Build a Chatbot for a Fashion Store 

Develop and implement a Python-based chatbot that dynamically generates responses using data  from PostgreSQL database. The chatbot should interact with a database containing two tables, detailed below. When a user asks a question (e.g., “Show me the available dresses under $50” or “How many red dresses were sold yesterday?”), the system should accurately retrieve and return the relevant information. 

Database Schema 
Table 1: dress_stocks 
● dress_id (Primary Key): Unique identifier for each dress. 
● brand
● dress_type: The type or category of the dress (e.g., casual, formal). 
● size
● colour
● material
● unit_price: Price per dress. 
● discount_percentage
● available_stock
● total_units: Number of dresses available in stock. 
Table 2: daily_sales 
● sale_id (Primary Key): Unique identifier for each sale. 
● sale_date: The date of the sale. 
● dress_id: Foreign key that references dress_stocks.dress_id. 
● customer_id
● customer_name
● payment_method
● quantity_sold: Number of dresses sold in that transaction. 
● sale_amount: Total amount for the sale. 
● discount_applied
● final_price

Requirements 
● Programming Language & Tools: 
● Use Python for building the chatbot. 
● Use PostgreSQL as the database system. 

Technology:
💻 Backend:
    Python → Main programming language for logic and processing
    Streamlit → UI framework for chatbot interaction
    LangChain → Handles LLM-based SQL query generation and execution
    PostgreSQL → Database system for storing and querying data
    SQLAlchemy → ORM for connecting with PostgreSQL

🤖 AI & LLM:
    Google Gemini 1.5 Pro → LLM for query generation & response
    LangChain Prompt Templates → Structured prompting for SQL query generation

🛠️ Tools & Utilities:
    dotenv → Load environment variables securely
    psycopg2 → PostgreSQL driver for Python
    LangGraph → Manages the query execution flow

🚀 Deployment & Version Control:
    Git & GitHub → Version control
    Streamlit Cloud / Docker (optional) → Deployment options

Queries
One-Table Queries (Using a Single Table)
○ Sales Data Queries (daily_sales.csv)
    1. "What was the total sales amount for January 2024?"
    2. "Which payment method was used the most?"
    3. "Find the top 3 customers who made the highest purchases."
    4. "What is the average discount applied on sales?"

○ Stock Data Queries (dress_stocks.csv)
    1. "Which dress brand has the highest available stock?"
    2. "List all available dress colors."
    3. "Find the most expensive dress based on unit price."
    4. "What is the average discount percentage for all dresses?"

Two-Table Queries (Using Both Tables)
    1. "What is the total revenue generated for each dress brand?"
    2. "Which dress type has the highest sales quantity?"
    3. "Find the top 5 most sold dresses along with their brands and total sales amount."
    4. "What is the remaining stock for each dress after deducting total sales?"


Evaluation Criteria: 
○ Correctness: How accurately the system translates natural language queries into SQL. 
○ Code Quality: Readability, structure, and maintainability of the code. 
○ NLP Implementation: Effectiveness of your approach in parsing and understanding natural language. 
○ Perplexity: measures how well a model predicts a given text.
○ Bleu score: Measures similarity between generated text and reference text (ground truth).
○ Accuracy: Measures how often the generated SQL query is correct.#   S Q L _ a g e n t i c _ c h a t b o t 
 
 
