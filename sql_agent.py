from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
import os
import logging
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDv5E0ApDmQ2YLAzCEEIKGxhR5G7DdyIY4"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
   
    # other params...
)

#Trace, debug and monitor your application
LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="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"
LANGSMITH_PROJECT="pr-artistic-driveway-30"
OPENAI_API_KEY="<your-openai-api-key>"


memory = MemorySaver()

# db = SQLDatabase.from_uri("sqlite:///shoppinglite.db")
# print(db.dialect)
# print(db.get_usable_table_names())
# print(db.run("SELECT ds.brand FROM dress_stocks AS ds JOIN daily_sales AS dls ON ds.dress_id = dls.dress_id WHERE dls.customer_name = 'Alice';")) ##What is the brand bought by Alice?

POSTGRES_URI = "postgresql+psycopg2://postgres:1837@localhost:5432/shoppingpost"
db = SQLDatabase.from_uri(POSTGRES_URI)


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

prompt_template = ChatPromptTemplate.from_template(
    "Given the following SQL tables, write a SQL query to answer the user's question.\n"
    "Tables:\n"
    "{schema}\n\n"
    "Question: {question}\n"
    "SQL Query:"
)

def write_query(state: State):
    """Generate SQL query to fetch information."""
    try:
        prompt = prompt_template.invoke({
            "schema": db.get_table_info(),
            "question": state["question"],
        })
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return {"query": result["query"]}
    except Exception as e:
        logging.error(f"Query generation error: {e}")
        return {"query": ""}  # Return empty query on failure


# write_query({"question": "How many quantity of clothes alice brought?"})


def execute_query(state: State):
    """Execute SQL query."""
    try:
        logging.info(f"Executing Query: {state['query']}")
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        result = execute_query_tool.invoke(state["query"])
        return {"result": result}
    except Exception as e:
        logging.error(f"SQL execution error: {e}")
        return {"result": "Query execution failed."}

# execute_query({"query": "SELECT COUNT(EmployeeId) AS EmployeeCount FROM Employee;"})

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    try:
        prompt = f"""
        Given the following user question, corresponding SQL query,
        and SQL result, answer the user question.
        
        Question: {state["question"]}
        SQL Query: {state["query"]}
        SQL Result: {state["result"]}
        """
        response = llm.invoke(prompt)
        return {"answer": response.content}
    except Exception as e:
        logging.error(f"Answer generation error: {e}")
        return {"answer": "Failed to generate an answer."}



graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()





for step in graph.stream(
    {"question": "Find the top 5 most sold dresses along  total sales amount."}, stream_mode="updates"
):
    if "generate_answer" in step:
        final_result = step["generate_answer"]["answer"]

if final_result:
    print(final_result)


# #Human-in-loop

# from langgraph.checkpoint.memory import MemorySaver

# memory = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory, interrupt_before=["execute_query"])

# # Now that we're using persistence, we need to specify a thread ID
# # so that we can continue the run after review.
# config = {"configurable": {"thread_id": "1"}}

# for step in graph.stream(
#     {"question": "Find the top 5 most sold dresses along  total sales amount."},
#     config,
#     stream_mode="updates",
# ):
#     print(step)

# try:
#     user_approval = input("Do you want to go to execute query? (yes/no): ")
# except Exception:
#     user_approval = "no"

# if user_approval.lower() == "yes":
#     # If approved, continue the graph execution
#     for step in graph.stream(None, config, stream_mode="updates"):
#         print(step)
# else:
#     print("Operation cancelled by user.")