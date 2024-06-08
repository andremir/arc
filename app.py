
import streamlit as st
from sqlalchemy import create_engine, text
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import matplotlib.pyplot as plt
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set page title and logo
st.set_page_config(page_title='ARC')
st.image("Arc-Logo.png", width=200)

# Define connection information for BigQuery
connection_info = {
    "project_id": st.secrets["gcp_service_account"]["project_id"],
    "dataset_id": "Arc1",  # Replace with your dataset name
    "table_name": "mkt",  # Replace with your table name
}

# Load the credentials from the Streamlit secrets
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

# Create API client for BigQuery
client = bigquery.Client(credentials=credentials)

# Perform query using BigQuery client
@st.cache_data(ttl=600)  # Cache results for 600 seconds (10 minutes)
def run_query(query):
    query_job = client.query(query)
    rows_raw = query_job.result()
    # Convert to list of dicts. Required for st.cache_data to hash the return value.
    rows = [dict(row) for row in rows_raw]
    return rows

# Query the schema of the table to get column names
schema_query = f"""
SELECT column_name
FROM `{connection_info['project_id']}.{connection_info['dataset_id']}.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = '{connection_info['table_name']}';
"""
columns_df = run_query(schema_query)
# st.write("Columns in the table:")
# st.write(pd.DataFrame(columns_df))

# Get data and create DataFrame
rows = run_query(f"SELECT * FROM `{connection_info['project_id']}.{connection_info['dataset_id']}.{connection_info['table_name']}`")
data = pd.DataFrame(rows)  

# Display the table
st.write("Entire BIG Query Content:")
st.dataframe(data)  

# --- Data Analysis Section ---
st.subheader('Data Analysis')

# Summary Statistics
st.write('Summary Statistics')
st.write(data.describe())

# --- Scatter Plot Section ---
st.subheader('Scatter Plot')
with st.expander("Scatter Plot Options"):  # Expandable section
    x_column = st.selectbox('Select X-axis column', data.columns)
    y_column = st.selectbox('Select Y-axis column', data.columns)

    # Create and display scatter plot
    fig, ax = plt.subplots(figsize=(10, 6)) # Reduced plot size
    ax.scatter(data[x_column], data[y_column])
    ax.set_xlabel(x_column, fontsize=12)
    ax.set_ylabel(y_column, fontsize=12)
    st.pyplot(fig)

# --- Histogram Section ---
st.subheader('Histogram')
with st.expander("Histogram Options"):
    column = st.selectbox('Select column', data.columns)

    # Create and display histogram
    fig, ax = plt.subplots(figsize=(10, 6)) 
    ax.hist(data[column])
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    st.pyplot(fig)

# Initialize the language model with the OpenAI API key from secrets
openai_api_key = st.secrets["openai"]["api_key"]
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=openai_api_key
)

# Create a mapping of natural language terms to actual column names
column_mapping = {
    "month": "Month",
    "ad impressions": "Ad_Impressions",
    "clicks": "Clicks",
    "conversions": "Conversions",
    "cost": "Cost",
    "revenue": "Revenue",
    "ctr": "CTR",
    "cpc": "CPC",
    "conversion rate": "Conversion_Rate",
    "roas": "ROAS"
}

# Convert column mapping to a string format for the prompt
column_mapping_str = ", ".join([f"'{k}': '{v}'" for k, v in column_mapping.items()])

# Define the prompt template for the LLM to generate SQL queries
prompt = PromptTemplate(
    input_variables=["query", "project", "dataset", "table", "column_mapping"],
    template="""
    Given the following natural language query and a mapping of terms to column names, generate a SQL query for BigQuery using the fully qualified table name including the project, dataset, and table name:
    Project: {project}
    Dataset: {dataset}
    Table: {table}
    Column Mapping: {column_mapping}
    Query: {query}
    SQL Query:
    """
)

# Create an LLM chain with the prompt template and language model
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Function to ask a new question and get the result
def ask_question(natural_language_query: str):
    # Generate the SQL query using the LLM
    sql_query = llm_chain.run({
        "query": natural_language_query,
        "project": connection_info["project_id"],
        "dataset": connection_info["dataset_id"],
        "table": connection_info["table_name"],
        "column_mapping": column_mapping_str
    })

    # Print the generated SQL query
    st.write(f"Generated SQL Query: {sql_query}")

    # Query the database and print the results
    result_df = run_query(sql_query)
    st.write(result_df)
    return result_df

# --- Natural Language Query Section ---
st.subheader("Natural Language Query Interface")
user_query = st.text_input("Enter your query:")

if st.button("Submit Query"):
    if user_query:
        ask_question(user_query)
    else:
        st.write("Please enter a query.")

# --- Contact Section ---
st.header('Contact')
st.write('For any inquiries, please contact us at xxxx@gmail.com.')
