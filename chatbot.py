# from langchain_community.utilities import SQLDatabase
# from typing_extensions import TypedDict
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from typing_extensions import Annotated
# from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
# from langgraph.graph import START, StateGraph
# import os
# import streamlit as st
# from dotenv import load_dotenv
# from datetime import datetime
# from langgraph.checkpoint.memory import MemorySaver

# # Load environment variables
# load_dotenv()


# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = "AIzaSyDv5E0ApDmQ2YLAzCEEIKGxhR5G7DdyIY4"

# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     temperature=0,
   
#     # other params...
# )

# #Trace, debug and monitor your application
# LANGSMITH_TRACING=True
# LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
# LANGSMITH_API_KEY="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"
# LANGSMITH_PROJECT="pr-artistic-driveway-30"
# OPENAI_API_KEY="<your-openai-api-key>"

# today = datetime.now().strftime("%d/%m/%Y")

# memory = MemorySaver()

# db = SQLDatabase.from_uri("sqlite:///shoppinglite.db")
# # print(db.dialect)
# # print(db.get_usable_table_names())
# # print(db.run("SELECT ds.brand FROM dress_stocks AS ds JOIN daily_sales AS dls ON ds.dress_id = dls.dress_id WHERE dls.customer_name = 'Alice';")) ##What is the brand bought by Alice?


# class State(TypedDict):
#     question: str
#     query: str
#     result: str
#     answer: str


# class QueryOutput(TypedDict):
#     """Generated SQL query."""

#     query: Annotated[str, ..., "Syntactically valid SQL query."]

# prompt_template = ChatPromptTemplate.from_template(
#     "Given the following SQL tables, write a SQL query to answer the user's question.\n"
#     "Tables:\n"
#     "{schema}\n\n"
#     "Question: {question}\n"
#     "SQL Query:"
# )

# def write_query(state: State):
#     """Generate SQL query to fetch information."""
#     prompt = prompt_template.invoke(
#         {
#             "schema": db.get_table_info(),
#             "question": state["question"],
#         }
#     )
#     # structured_llm = llm.with_structured_output(QueryOutput).invoke(prompt)
#     # return {"query": structured_llm["query"]}
#     structured_llm = llm.with_structured_output(QueryOutput)
#     result = structured_llm.invoke(prompt)
#     return {"query": result["query"]}


# # write_query({"question": "How many quantity of clothes alice brought?"})


# def execute_query(state: State):
#     """Execute SQL query."""
#     execute_query_tool = QuerySQLDatabaseTool(db=db)
#     return {"result": execute_query_tool.invoke(state["query"])}

# # execute_query({"query": "SELECT COUNT(EmployeeId) AS EmployeeCount FROM Employee;"})

# def generate_answer(state: State):
#     """Answer question using retrieved information as context."""
#     prompt = (
#         "Given the following user question, corresponding SQL query, "
#         "and SQL result, answer the user question.\n\n"
#         f'Question: {state["question"]}\n'
#         f'SQL Query: {state["query"]}\n'
#         f'SQL Result: {state["result"]}'
#     )
#     response = llm.invoke(prompt)
#     return {"answer": response.content}


# graph_builder = StateGraph(State).add_sequence(
#     [write_query, execute_query, generate_answer]
# )
# graph_builder.add_edge(START, "write_query")
# graph = graph_builder.compile()



# # Streamlit UI for Chatbot
# st.title("🛍️ SQL Query Chatbot")


# # User Input
# user_input = st.chat_input("Ask me a question about the database...")

# with st.sidebar:
#         st.title("Banking Assistant Settings")
#         st.markdown(f"**Current Date:** {today}")
#         st.divider()
        
#         if st.button("Clear Conversation"):
#             st.session_state.messages = []
#             st.rerun()

# # Initialize session state
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "tool_used" not in st.session_state:
#         st.session_state.tool_used = None

#     # Display chat history with avatars
#     for message in st.session_state.messages:
#         avatar = "👤" if message["role"] == "user" else "🏦"
#         with st.chat_message(message["role"], avatar=avatar):
#             st.markdown(message["content"])
#             if message["role"] == "assistant" and st.session_state.tool_used:
#                 st.caption(f"Retrieved using: {st.session_state.tool_used}")
#                 st.session_state.tool_used = None

#     # User input with processing indicator
#     if query := st.chat_input("Ask your banking question..."):
#         st.session_state.messages.append({"role": "user", "content": query})
        
#         # Display user message immediately
#         with st.chat_message("user", avatar="👤"):
#             st.markdown(query)
        
#         # Create a placeholder for the assistant's response
#         with st.chat_message("assistant", avatar="🏦"):
#             response_placeholder = st.empty()
#             with st.spinner("Processing your request..."):
#                 try:
#                     start_time = time.time()
                    
#                     # Process the query
#                     result_state = app.invoke(
#                         {"messages": [{"role": "user", "content": query}]},
#                         config={"configurable": {"thread_id": "7"}},
#                     )
#                     response = result_state["messages"][-1].content
                    
#                     # Simulate typing effect
#                     full_response = ""
#                     for chunk in response.split():
#                         full_response += chunk + " "
#                         time.sleep(0.05)
#                         response_placeholder.markdown(full_response + "▌")
                    
#                     response_placeholder.markdown(full_response)
#                     st.session_state.messages.append({"role": "assistant", "content": response})
                    
#                     # Performance metrics
#                     processing_time = time.time() - start_time
#                     st.caption(f"Processed in {processing_time:.2f} seconds")
                    
#                 except Exception as e:
#                     error_message = "Sorry, I encountered an error processing your request. Please try again later."
#                     response_placeholder.error(error_message)
#                     st.session_state.messages.append({"role": "assistant", "content": error_message})
#                     st.error(f"Error details: {str(e)}")

# if __name__ == "__main__":
#     main()

# if user_input:
#     final_result = "I'm sorry, but I couldn't generate an answer."

#     for step in graph.stream({"question": user_input}, stream_mode="updates"):
#         if "generate_answer" in step:
#             final_result = step["generate_answer"]["answer"]

#     st.write(final_result)


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
db = SQLDatabase.from_uri("sqlite:///shoppinglite.db")

today = datetime.now().strftime("%d/%m/%Y")

# Define state structure
class State(TypedDict):
    question: str
    query: str
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

def write_query(state: State):
    prompt = prompt_template.invoke(
        {"schema": db.get_table_info(), "question": state["question"]}
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def execute_query(state: State):
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State):
    prompt = (
        f"Given the following user question, corresponding SQL query, "
        f"and SQL result, answer the user question.\n\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

# Build execution graph
graph_builder = StateGraph(State)
graph_builder.add_node("write_query", write_query)
graph_builder.add_node("execute_query", execute_query)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_edge(START, "write_query")
graph_builder.add_edge("write_query", "execute_query")
graph_builder.add_edge("execute_query", "generate_answer")
graph = graph_builder.compile()

# Streamlit UI
st.title("🛍️ SQL Query Chatbot")

with st.sidebar:
    
    st.sidebar.image("docs/images.png", caption="SQL Chatbot", width=200,use_container_width=True)
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
                for step in graph.stream({"question": user_input}):
                    if "generate_answer" in step:
                        final_result = step["generate_answer"]["answer"]
                
                response_placeholder.markdown(final_result)
                st.session_state.messages.append({"role": "assistant", "content": final_result})
                processing_time = time.time() - start_time
                st.caption(f"Processed in {processing_time:.2f} seconds")
            except Exception as e:
                error_message = "Sorry, an error occurred. Please try again."
                response_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.error(f"Error details: {str(e)}")
