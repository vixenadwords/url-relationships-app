import streamlit as st
import pandas as pd
import numpy as np
import openai

# Function to get embeddings from OpenAI
def get_embeddings(text, api_key):
    openai.api_key = api_key
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'])

# Streamlit app
st.title("URL Relationships App")

# API Key input
api_key = st.text_input("Enter your OpenAI API key", type="password")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file and api_key:
    # Read file
    df = pd.read_csv(uploaded_file)
    st.write("File successfully uploaded!")

    # Display columns and let user map them
    url_column = st.selectbox("Select URL column", df.columns)
    embedding_column = st.selectbox("Select Embeddings column (if precomputed)", ["None"] + list(df.columns))

    # Allow user to input expected embedding length
    expected_length = st.number_input("Enter expected embedding length", min_value=1, value=1536)

    # Generate embeddings if not precomputed
    if embedding_column == "None":
        df['Embeddings'] = df[url_column].apply(lambda x: get_embeddings(x, api_key))
    else:
        df['Embeddings'] = df[embedding_column].apply(lambda x: np.array([float(i) for i in str(x).split(',')]))

    # Filter correct shape
    filtered_df = df[df['Embeddings'].apply(lambda x: len(x) == expected_length)]

    # Display filtered dataframe
    st.write("Filtered DataFrame", filtered_df.head())

    # Further processing and visualization...
    # Placeholder for additional processing and visualization steps
    st.write("Further processing and visualization steps will go here.")


