import os
import time
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict, Annotated

# Load environment variables
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDv5E0ApDmQ2YLAzCEEIKGxhR5G7DdyIY4"

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

# Initialize memory and database
memory = MemorySaver()
DATABASE_URL = "postgresql+psycopg2://postgres:1837@localhost:5432/shoppingpost"
db = SQLDatabase.from_uri(DATABASE_URL)
# db = SQLDatabase.from_uri("sqlite:///shoppinglite.db")

today = datetime.now().strftime("%d/%m/%Y")

# Define state structure
class State(TypedDict):
    question: str
    query: str
    validated_query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

# Define prompts
prompt_template = ChatPromptTemplate.from_template(
    "Given the following SQL tables, write a SQL query to answer the user's question.\n"
    "Tables:\n"
    "{schema}\n\n"
    "Question: {question}\n"
    "SQL Query:"
)

# Define query validation system message
query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.
"""

# Create validation chain
query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("human", "{query}")]
)
validation_chain = query_check_prompt | llm | StrOutputParser()

def write_query(state: State):
    """Generate SQL query from the user's question."""
    prompt = prompt_template.invoke(
        {"schema": db.get_table_info(), "question": state["question"]}
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def validate_query(state: State):
    """Validate and refine the generated SQL query."""
    validated_query = validation_chain.invoke({"query": state["query"]}).strip()
    validated_query = validated_query.replace("```sql", "").replace("```", "").strip()
    return {"validated_query": validated_query}

def execute_query(state: State):
    """Execute the validated SQL query and return results."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["validated_query"])}

def generate_answer(state: State):
    """Generate a human-readable answer based on SQL results."""
    prompt = (
        f"Given the following user question, corresponding SQL query, "
        f"and SQL result, answer the user question.\n\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['validated_query']}\n"
        f"SQL Result: {state['result']}"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

# Build execution graph
graph_builder = StateGraph(State)
graph_builder.add_node("write_query", write_query)
graph_builder.add_node("validate_query", validate_query)
graph_builder.add_node("execute_query", execute_query)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "write_query")
graph_builder.add_edge("write_query", "validate_query")
graph_builder.add_edge("validate_query", "execute_query")
graph_builder.add_edge("execute_query", "generate_answer")

graph = graph_builder.compile()

# Streamlit UI
st.title("🛍️ SQL Query Chatbot")

with st.sidebar:
    st.sidebar.image("docs/images.png", caption="SQL Chatbot", width=200, use_container_width=True)
    st.title("Chatbot Settings")
    st.markdown(f"**Current Date:** {today}")
    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask me a question about the database...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        with st.spinner("Processing your request..."):
            try:
                start_time = time.time()
                final_result = "I'm sorry, but I couldn't generate an answer."
                
                step_output = graph.invoke({"question": user_input})

                validated_query = step_output["validated_query"]
                result = step_output["result"]
                final_result = step_output["answer"]

                # Display validated SQL query
                st.markdown("**Validated SQL Query:**")
                st.code(validated_query, language="sql")

                # Display SQL query results
                st.markdown("**SQL Query Result:**")
                if isinstance(result, list):
                    st.dataframe(result)
                else:
                    st.text(result)

                # Display final answer
                response_placeholder.markdown(final_result)
                st.session_state.messages.append({"role": "assistant", "content": final_result})

                processing_time = time.time() - start_time
                st.caption(f"Processed in {processing_time:.2f} seconds")
            except Exception as e:
                error_message = "Sorry, an error occurred. Please try again."
                response_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.error(f"Error details: {str(e)}")
