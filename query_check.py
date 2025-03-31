# from langchain_community.utilities import SQLDatabase
# from typing_extensions import TypedDict
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from typing_extensions import Annotated
# from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
# from langgraph.graph import START, StateGraph
# import os
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# from langchain.chains import create_sql_query_chain


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




# db = SQLDatabase.from_uri("sqlite:///shoppinglite.db")
# # print(db.dialect)
# # print(db.get_usable_table_names())
# # print(db.run("SELECT ds.brand FROM dress_stocks AS ds JOIN daily_sales AS dls ON ds.dress_id = dls.dress_id WHERE dls.customer_name = 'Alice';")) ##What is the brand bought by Alice?



# chain = create_sql_query_chain(llm, db)

# context = db.get_context()
# print(list(context))
# print(context["table_info"])



# # Define a system message to check SQL queries for common mistakes
# query_check_system = """You are a SQL expert with a strong attention to detail.
# Double check the SQLite query for common mistakes, including:
# - Using NOT IN with NULL values
# - Using UNION when UNION ALL should have been used
# - Using BETWEEN for exclusive ranges
# - Data type mismatch in predicates
# - Properly quoting identifiers
# - Using the correct number of arguments for functions
# - Casting to the correct data type
# - Using the proper columns for joins

# If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

# You will call the appropriate tool to execute the query after running this check."""

# # Create the prompt
# query_check_prompt = ChatPromptTemplate.from_messages(
#     [("system", query_check_system), ("human", "{query}")]
# )
# validation_chain = query_check_prompt | llm | StrOutputParser()

# full_chain = {"query": chain} | validation_chain

# query = full_chain.invoke(
#     {
#         "question": "Find the top 5 most sold dresses along  total sales amount."
#     }
# )
# query = query.strip().replace("```sql", "").replace("```", "")
# print(query)
# print(db.run(query))

from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API Keys
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDv5E0ApDmQ2YLAzCEEIKGxhR5G7DdyIY4"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

# Enable tracing & monitoring
LANGSMITH_TRACING = True
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_API_KEY = "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"
LANGSMITH_PROJECT = "pr-artistic-driveway-30"
OPENAI_API_KEY = "<your-openai-api-key>"

# Connect to the SQLite database
db = SQLDatabase.from_uri("sqlite:///shoppinglite.db")

# Define the state class
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

# Prompt template for SQL generation
prompt_template = ChatPromptTemplate.from_template(
    "Given the following SQL tables, write a SQL query to answer the user's question.\n"
    "Tables:\n"
    "{schema}\n\n"
    "Question: {question}\n"
    "SQL Query:"
)

def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = prompt_template.invoke(
        {
            "schema": db.get_table_info(),
            "question": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

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

# Define the graph workflow
graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

# Run the workflow to generate a query
question = "Find the top 5 most sold dresses along with total sales amount."
query_state = graph.invoke({"question": question})
sql_query = query_state["query"]  # Extract the generated SQL query
print("\nSQL query: ",sql_query)

# Validate the SQL query before execution
validated_query = validation_chain.invoke({"query": sql_query}).strip()

# Ensure proper SQL formatting
validated_query = validated_query.replace("```sql", "").replace("```", "").strip()

print("\nValidated SQL Query:")
print(validated_query)

# Execute the validated query
try:
    result = db.run(validated_query)
    print("\nQuery Result:", result)
except Exception as e:
    print("\nError executing query:", e)
