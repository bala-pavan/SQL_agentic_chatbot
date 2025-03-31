
import getpass
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate, PromptTemplate
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCHWHtn5uR1thWsHH_2rAlArqzFCTeV9qw"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
   
    # other params...
)

files=["docs\\daily_sales.csv", "docs\\dress_stocks.csv"]

engine=create_engine("sqlite:///docs/shopping.db")

for file in files:
    table_name=os.path.splitext(os.path.basename(file))[0]
    df=pd.read_csv(file)
    df.to_sql(table_name, engine, index=False, if_exists='replace')

db=SQLDatabase(engine=engine)

def get_table_info():
    """Get the schema of the database."""
    db_schema = db.get_table_info()
    return db_schema
 
db_schema = get_table_info()
 
prompt_template = PromptTemplate.from_template(
    "You are a SQL expert and you are capable of writing SQL queries to answer the user's question.\n"
    "Given the following SQL tables"
    "- Identify the relationships between the tables based on PK and FK relatioinships. Use your expertise to do that"
    "- generate SQL queries to answer the user's question.\n"
    "- If the external party id is provided in the question, use it to filter the results.\n"
    "Follow the below instructions as well:\n"
    "- CR stands for credit transactions and DB stands for debit transactions.\n"
    "Schema:\n"
    f"{db_schema}\n\n"
    "{input}\n\n"
    "{agent_scratchpad}"
    "execute the SQL query and return the results in a structured format.\n"
)
 
sql_agent = create_sql_agent(llm,
                            db=db,
                            agent_type="tool-calling",
                            verbose=True,
                            max_iterations=25,
                            prompt=prompt_template,
                            )
 
def get_sql_agent():
    """returns the sql agent."""
    return sql_agent
 
def get_customer_details(query:str) -> str:
    """Get the details of a customer."""
    print("Inside get_customer_details SQL Agent", query)
    response = sql_agent.invoke({"input": query})
    return response["output"]

    
query="What is product and brand that brought by Isaac?"
response=get_customer_details(query)
print(response)



