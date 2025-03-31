
import re
import os
import asyncio
import pandas as pd
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Float,
    Integer,
    Date,
    ForeignKey,
    insert,
    select,
)
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime

# Load environment variables
load_dotenv()

# Ensure GOOGLE_API_KEY is set
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCHWHtn5uR1thWsHH_2rAlArqzFCTeV9qw"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

def create_database():
    """
    Creates and populates an SQLite database with tables for dress_stocks and daily_sales.
    Returns the SQLDatabase object and table references.
    """
    engine = create_engine("sqlite:///shoppinglite.db")
    metadata_obj = MetaData()

    # Define dress_stocks table
    dress_stocks_table = Table(
        "dress_stocks",
        metadata_obj,
        Column("dress_id", Integer, primary_key=True),
        Column("brand", String(50)),
        Column("dress_type", String(50), nullable=False),
        Column("size", String(10)),
        Column("color", String(30)),
        Column("material", String(50)),
        Column("unit_price", Float, nullable=False),
        Column("discount_percentage", Integer, nullable=False),
        Column("available_stock", Integer, nullable=False),
    )

    # Define daily_sales table
    daily_sales_table = Table(
        "daily_sales",
        metadata_obj,
        Column("sale_id", Integer, primary_key=True),
        Column("sale_date", Date, nullable=False),
        Column("dress_id", Integer, ForeignKey("dress_stocks.dress_id"), nullable=False),
        Column("customer_id", Integer, nullable=False),
        Column("customer_name", String(30), nullable=False),
        Column("payment_method", String(30), nullable=False),
        Column("quantity_sold", Integer, nullable=False),
        Column("sale_amount", Float, nullable=False),
        Column("discount_applied", Float, nullable=False),
        Column("final_price", Float, nullable=False),
    )

    # Reset Database
    metadata_obj.drop_all(engine)
    metadata_obj.create_all(engine)

    return engine, dress_stocks_table, daily_sales_table


def load_csv(file_path, table, engine, unique_key=None):
    """
    Loads CSV data into the specified SQL table.
    """
    try:
        df = pd.read_csv(file_path, quotechar='"', escapechar="\\")
        df.columns = df.columns.str.strip()

        # Ensure unique key exists in DataFrame
        if unique_key and unique_key not in df.columns:
            raise KeyError(f"Column '{unique_key}' not found in {file_path}.")

        if unique_key:
            df = df.drop_duplicates(subset=[unique_key])

        df = df.fillna("NULL")

        # Convert 'sale_date' to datetime if present
        if "sale_date" in df.columns:
            df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce").dt.date

        with engine.begin() as connection:
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict = {k: (None if v == "NULL" else v) for k, v in row_dict.items()}

                if unique_key:
                    exists = connection.execute(
                        select(table).where(table.c[unique_key] == row_dict[unique_key])
                    ).fetchone()
                    if exists:
                        print(f"Skipping duplicate: {row_dict[unique_key]}")
                        continue

                try:
                    stmt = insert(table).values(**row_dict)
                    connection.execute(stmt)
                except IntegrityError as e:
                    print(f"Integrity Error: {e} | Row: {row_dict}")
                except Exception as e:
                    print(f"Error inserting row: {e} | Row: {row_dict}")

        print(f"Successfully loaded data from {file_path}")

    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Skipping...")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")


# Create Database & Load Data
engine, dress_stocks_table, daily_sales_table = create_database()
load_csv("docs/dress_stocks.csv", dress_stocks_table, engine, unique_key="dress_id")
load_csv("docs/daily_sales.csv", daily_sales_table, engine, unique_key="sale_id")

# Create SQLDatabase object
sql_database = SQLDatabase(engine, include_tables=["dress_stocks", "daily_sales"])

# Define Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Given the following SQL tables, write a SQL query to answer the user's question.\n"
    "Tables:\n"
    "{schema}\n\n"
    "Question: {question}\n"
    "SQL Query:"
)

# Define LCEL Chain
chain = (
    {"schema": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def extract_sql_query(response: str) -> str:
    """
    Extracts the SQL query from the LLM's response by removing Markdown code blocks.
    """
    match = re.search(r"```sql(.*?)```", response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()


    
def execute_query(natural_language_query: str) -> str:
    try:
        schema = sql_database.get_table_info()
        # print("Database Schema:\n", schema)  # Debugging step
        llm_response = chain.invoke({"schema": schema, "question": natural_language_query})
        sql_query = extract_sql_query(llm_response)
        
        print(f"Generated SQL Query: {sql_query}")  # Debugging step
        
        # Run the corrected SQL query
        result = sql_database.run(sql_query)
        return result
    except Exception as e:
        return f"An error occurred: {str(e)}"



# Async Query Execution
async def process(query: str) -> str:
    return await asyncio.to_thread(execute_query, query)


# Testing the System
if __name__ == "__main__":
    test_query = "What brand of dress did Alice buy?"
    result = asyncio.run(process(test_query))
    print("SQL Agent Response:\n", result)
