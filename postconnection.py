from sqlalchemy import create_engine
import pandas as pd
import os

# PostgreSQL connection string
DATABASE_URL = "postgresql+psycopg2://postgres:password@localhost:5432/shoppingpost"


# Create engine
engine = create_engine(DATABASE_URL)

# List of CSV files
files = ["docs/daily_sales.csv", "docs/dress_stocks.csv"]

# Loop through files and upload them
for file in files:
    table_name = os.path.splitext(os.path.basename(file))[0]  # Extract table name from filename
    df = pd.read_csv(file)  # Read CSV
    df.to_sql(table_name, engine, index=False, if_exists="replace")  # Upload to PostgreSQL

print("CSV files uploaded successfully!")
