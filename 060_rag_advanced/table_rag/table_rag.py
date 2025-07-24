#%%
import sqlite3
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

#%% Pydantic SQL-Query Model
class SQLQuery(BaseModel):
    sql_query: str = Field(description="The SQL query to answer the user question.")

#%%
def fetch_information_from_db(query: str):
    conn = sqlite3.connect('coffee_sales.db')
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        res = cursor.fetchall()
        conn.close()
        return res
    except Exception as e:
        conn.close()
        print(f"Error executing query: {e}")
        return []

# SQLite doesn't support SHOW TABLES. Instead, query the sqlite_master table
user_query = "SELECT name FROM sqlite_master WHERE type='table';"  # lists all tables
fetch_information_from_db(user_query)
#%%
def create_sql_query(user_query: str, sql_table_info: str):
    model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a SQL expert and can create a SQL query to answer the user's question. You will be given a SQL table information and a user question. You will need to create a SQL query to answer the user question. You will return a JSON with an object sql_query"),
            ("user", "SQL table information: {sql_table_info} \n User question: {user_query}"),
        ]
    )
    chain = prompt | model | JsonOutputParser(pydantic_object=SQLQuery)
    return chain.invoke({"sql_table_info": sql_table_info, "user_query": user_query})


# %%
sql_table_info = """
CREATE TABLE coffee_sales (
    sale_id INT PRIMARY KEY,
    product_name VARCHAR(100),
    origin_country VARCHAR(50),
    bean_type VARCHAR(50),
    roast_level VARCHAR(20),
    price_per_kg DECIMAL(6,2),
    quantity_kg DECIMAL(6,2),
    sale_date DATE,
    customer_id INT,
    region VARCHAR(50),
    organic BOOLEAN,
    certification VARCHAR(50)
);
"""
user_query = "What is the total sales over all time?"
sql_query = create_sql_query(user_query, sql_table_info)
fetch_information_from_db(query=sql_query['sql_query'])
# %% query that was created
sql_query['sql_query']

#%%
def rag(user_query: str, sql_table_info: str):
    sql_query = create_sql_query(user_query, sql_table_info)
    print(sql_query['sql_query'])
    retrieved_information = fetch_information_from_db(query=sql_query['sql_query'])
    print(f"Retrieved Info: {retrieved_information}")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    # prepare chain
    messages = [
        ("system", """
        You are an expert on coffee and you can answer questions based on data from the table. please answer purely based on the retrieved information.
        Here is the information from the database call {retrieved_information}.
        The underlying SQL-query is {sql_query}.
        """),
        ("user", "{user_query}")
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    chain = prompt_template | llm | StrOutputParser()
    # invoke chain
    model_response = chain.invoke({"retrieved_information": retrieved_information, "user_query": user_query, "sql_query": sql_query['sql_query']})
    return model_response

#%% testing
user_query = "What's the average price of organic coffee compared to non-organic?"
# user_query = "How does the price depend on the roast level?"
res = rag(user_query, sql_table_info)
res

# %%
