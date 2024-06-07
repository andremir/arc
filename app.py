


# # # Upload CSV file
# # uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

# # if uploaded_file is not None:
# #     # Read CSV file
# #     data = pd.read_csv(uploaded_file)

# #     # Display data
# #     st.subheader('Data Preview')
# #     st.write(data.head())

# #     # Data analysis and visualization
# #     st.subheader('Data Analysis')

# #     # Display summary statistics
# #     st.write('Summary Statistics')
# #     st.write(data.describe())

# #     # Select columns for scatter plot
# #     st.write('Scatter Plot')
# #     x_column = st.selectbox('Select X-axis column', data.columns)
# #     y_column = st.selectbox('Select Y-axis column', data.columns)

# #     # Create scatter plot
# #     fig, ax = plt.subplots(figsize=(20, 8))
# #     ax.scatter(data[x_column], data[y_column])
# #     ax.set_xlabel(x_column,fontsize=14)
# #     ax.set_ylabel(y_column,fontsize=14)
# #     ax.tick_params(axis='both', labelsize=14)
# #     st.pyplot(fig)

# #     # Select column for histogram
# #     st.write('Histogram')
# #     column = st.selectbox('Select column', data.columns)

# #     # Create histogram
# #     fig, ax = plt.subplots(figsize=(20, 8))
# #     ax.hist(data[column])
# #     ax.set_xlabel(column)
# #     ax.set_ylabel('Frequency')
# #     ax.tick_params(axis='both', labelsize=14)
# #     st.pyplot(fig)

# # # Additional sections and features

# # st.header('Contact')
# # st.write('For any inquiries, please contact us at andremi@gmail.com.')
# # streamlit_app.py

# import streamlit as st
# from google.oauth2 import service_account
# from google.cloud import bigquery
# import pandas as pd
# import matplotlib.pyplot as plt

# # Set page title
# st.set_page_config(page_title='ARC')

# # Set Logo
# st.image("Arc-Logo.png", width=200)

# # Create API client.
# credentials = service_account.Credentials.from_service_account_info(
#     st.secrets["gcp_service_account"]
# )
# client = bigquery.Client(credentials=credentials)

# # Perform query.
# # Uses st.cache_data to only rerun when the query changes or after 10 min.
# @st.cache_data(ttl=600)
# def run_query(query):
#     query_job = client.query(query)
#     rows_raw = query_job.result()
#     # Convert to list of dicts. Required for st.cache_data to hash the return value.
#     rows = [dict(row) for row in rows_raw]
#     return rows

# rows = run_query("SELECT * FROM `arc1-425701.Arc1.mkt` LIMIT 100")    

# # Display the table
# st.write("Entire BIG Query Content:")
# st.dataframe(rows)  # This will display the table in a nice format

# # Data analysis and visualization
# st.subheader('Data Analysis')

# # Display summary statistics
# st.write('Summary Statistics')
# st.write(data.describe())

# # Select columns for scatter plot
# st.write('Scatter Plot')
# x_column = st.selectbox('Select X-axis column', data.columns)
# y_column = st.selectbox('Select Y-axis column', data.columns)

# # Create scatter plot
# fig, ax = plt.subplots(figsize=(20, 8))
# ax.scatter(data[x_column], data[y_column])
# ax.set_xlabel(x_column,fontsize=14)
# ax.set_ylabel(y_column,fontsize=14)
# ax.tick_params(axis='both', labelsize=14)
# st.pyplot(fig)

# # Select column for histogram
# st.write('Histogram')
# column = st.selectbox('Select column', data.columns)

# # Create histogram
# fig, ax = plt.subplots(figsize=(20, 8))
# ax.hist(data[column])
# ax.set_xlabel(column)
# ax.set_ylabel('Frequency')
# ax.tick_params(axis='both', labelsize=14)
# st.pyplot(fig)

# # Additional sections and features

# st.header('Contact')
# st.write('For any inquiries, please contact us at andremi@gmail.com.')


import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import matplotlib.pyplot as plt

# Set page title
st.set_page_config(page_title='ARC')

# Set Logo
st.image("Arc-Logo.png", width=200)

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

# Perform query.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)  # Cache results for 600 seconds (10 minutes)
def run_query(query):
    query_job = client.query(query)
    rows_raw = query_job.result()
    # Convert to list of dicts. Required for st.cache_data to hash the return value.
    rows = [dict(row) for row in rows_raw]
    return rows

# Get data and create DataFrame
rows = run_query("SELECT * FROM `arc1-425701.Arc1.mkt` LIMIT 100")
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

# --- Contact Section ---
st.header('Contact')
st.write('For any inquiries, please contact us at andremi@gmail.com.')



