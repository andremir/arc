

import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import re

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
client = bigquery.Client(credentials=credentials, project=connection_info["project_id"])

# Perform query using BigQuery client
@st.cache_data(ttl=600)  # Cache results for 600 seconds (10 minutes)
def run_query(query):
    try:
        query_job = client.query(query)
        rows_raw = query_job.result()
        rows = [dict(row) for row in rows_raw]
        return rows
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

# Query the schema of the table to get column names
schema_query = f"""
SELECT column_name
FROM `{connection_info['project_id']}.{connection_info['dataset_id']}.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = '{connection_info['table_name']}';
"""
columns_df = run_query(schema_query)

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

# --- Visualization Section ---
st.subheader('Visualizations')
with st.expander("Visualization Options"):  # Expandable section
    plot_type = st.selectbox('Select plot type', ['Scatter Plot', 'Histogram', 'Line Chart', 'Bar Chart', 'Heatmap'])
    
    if plot_type == 'Scatter Plot':
        x_column = st.selectbox('Select X-axis column', data.columns)
        y_column = st.selectbox('Select Y-axis column', data.columns)
        fig = px.scatter(data, x=x_column, y=y_column)
        st.plotly_chart(fig)

    elif plot_type == 'Histogram':
        column = st.selectbox('Select column', data.columns)
        fig = px.histogram(data, x=column)
        st.plotly_chart(fig)

    elif plot_type == 'Line Chart':
        x_column = st.selectbox('Select X-axis column', data.columns)
        y_column = st.selectbox('Select Y-axis column', data.columns)
        fig = px.line(data, x=x_column, y=y_column)
        st.plotly_chart(fig)

    elif plot_type == 'Bar Chart':
        x_column = st.selectbox('Select X-axis column', data.columns)
        y_column = st.selectbox('Select Y-axis column', data.columns)
        fig = px.bar(data, x=x_column, y=y_column)
        st.plotly_chart(fig)

    elif plot_type == 'Heatmap':
        if data.select_dtypes(include=['number']).shape[1] < 2:
            st.write("Heatmap requires at least two numerical columns.")
        else:
            fig = ff.create_annotated_heatmap(
                z=data.corr().values,
                x=list(data.corr().columns),
                y=list(data.corr().index),
                annotation_text=data.corr().round(2).values,
                showscale=True)
            st.plotly_chart(fig)

# Initialize the language model with the OpenAI API key from secrets
openai_api_key = st.secrets["openai"]["api_key"]
llm = ChatOpenAI(
    model="gpt-4",
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
sql_prompt = PromptTemplate(
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

# Define the prompt template for the LLM to generate natural language explanations and plot instructions
explanation_prompt = PromptTemplate(
    input_variables=["result"],
    template="""
    Given the following data result from a SQL query, provide a natural language summary of the result and instructions for plotting the data:
    Data Result: {result}
    Natural Language Explanation and Plot Instructions:
    """
)

# Create an LLM chain with the prompt template and language model for SQL query generation
sql_llm_chain = LLMChain(prompt=sql_prompt, llm=llm)

# Create an LLM chain with the prompt template and language model for natural language explanation and plotting instructions
explanation_llm_chain = LLMChain(prompt=explanation_prompt, llm=llm)

# Function to ask a new question and get the result
def ask_question(natural_language_query: str):
    # Generate the SQL query using the LLM
    sql_query = sql_llm_chain.run({
        "query": natural_language_query,
        "project": connection_info["project_id"],
        "dataset": connection_info["dataset_id"],
        "table": connection_info["table_name"],
        "column_mapping": column_mapping_str
    })

    # Print the generated SQL query
    st.write(f"Generated SQL Query: {sql_query}")

    # Query the database and get the results
    result_df = run_query(sql_query)
    if result_df:
        st.write("SQL Query Result:")
        result_df = pd.DataFrame(result_df)
        st.write(result_df)

        # Generate a natural language explanation of the results and plotting instructions
        explanation = explanation_llm_chain.run({
            "result": result_df.to_json(orient='split')
        })

        # Extracting the plot type and columns from the explanation
        st.write("Natural Language Explanation and Plot Instructions:")
        st.write(explanation)

        if "scatter" in explanation.lower():
            x_col, y_col = extract_columns(explanation, "scatter")
            fig = px.scatter(result_df, x=x_col, y=y_col)
            st.plotly_chart(fig)
        elif "histogram" in explanation.lower():
            col = extract_columns(explanation, "histogram")
            fig = px.histogram(result_df, x=col)
            st.plotly_chart(fig)
        elif "line" in explanation.lower():
            x_col, y_col = extract_columns(explanation, "line")
            fig = px.line(result_df, x=x_col, y=y_col)
            st.plotly_chart(fig)
        elif "bar" in explanation.lower():
            x_col, y_col = extract_columns(explanation, "bar")
            fig = px.bar(result_df, x=x_col, y=y_col)
            st.plotly_chart(fig)
        elif "heatmap" in explanation.lower():
            fig = ff.create_annotated_heatmap(
                z=result_df.corr().values,
                x=list(result_df.corr().columns),
                y=list(result_df.corr().index),
                annotation_text=result_df.corr().round(2).values,
                showscale=True)
            st.plotly_chart(fig)

def extract_columns(explanation, plot_type):
    # Function to extract columns from the LLM's explanation
    if plot_type in ["scatter", "line", "bar"]:
        match = re.search(r'X-axis column: (\w+), Y-axis column: (\w+)', explanation)
        if match:
            return match.group(1), match.group(2)
    elif plot_type == "histogram":
        match = re.search(r'column: (\w+)', explanation)
        if match:
            return match.group(1)
    return None, None

# --- Natural Language Query Section ---
st.subheader("Natural Language Query Interface")
user_query = st.text_input("Enter your query:", key="nl_query")

# Automatically submit the query when the user presses Enter
if user_query:
    ask_question(user_query)

# --- Contact Section ---
st.header('Contact')
st.write('For any inquiries, please contact us at xxxx@gmail.com.')

# --- Natural Language Query PandasAI Section ---
st.subheader("Natural Language Query PandasAI")
pandasai_query = st.text_input("Enter your query:", key="pandasai_query")

# Automatically submit the query when the user presses Enter
if pandasai_query:
    from pandasai import Agent
    # Configure PandasAI API Key
    pandas_api_key = st.secrets["PANDASAI"]["PANDAS_API_KEY"]

    # Set the environment variable
    os.environ["PANDASAI_API_KEY"] = pandas_api_key
    
    openai_api_key = st.secrets["openai"]["api_key"]
    agent = Agent(data)
    st.write(agent.chat(pandasai_query))
