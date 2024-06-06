import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set page title
st.set_page_config(page_title='ARC')

# Set Logo
st.image("Arc-Logo.png", width=200)

# Upload CSV file
uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file is not None:
    # Read CSV file
    data = pd.read_csv(uploaded_file)

    # Display data
    st.subheader('Data Preview')
    st.write(data.head())

    # Data analysis and visualization
    st.subheader('Data Analysis')

    # Display summary statistics
    st.write('Summary Statistics')
    st.write(data.describe())

    # Select columns for scatter plot
    st.write('Scatter Plot')
    x_column = st.selectbox('Select X-axis column', data.columns)
    y_column = st.selectbox('Select Y-axis column', data.columns)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.scatter(data[x_column], data[y_column])
    ax.set_xlabel(x_column,fontsize=14)
    ax.set_ylabel(y_column,fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    st.pyplot(fig)

    # Select column for histogram
    st.write('Histogram')
    column = st.selectbox('Select column', data.columns)

    # Create histogram
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.hist(data[column])
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='both', labelsize=14)
    st.pyplot(fig)

# Additional sections and features

st.header('Contact')
st.write('For any inquiries, please contact us at andremi@gmail.com.')