# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objs as go

# Title of the app
st.title('URL Relationships Visualization')

# Upload the file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Display the dataframe and column names
        st.write("Uploaded DataFrame:")
        st.write(df)

        # Allow the user to select the columns for URLs and Embeddings
        url_column = st.selectbox("Select the column for URLs", df.columns)
        embeddings_column = st.selectbox("Select the column for Embeddings", df.columns)

        # Provide a selection of common embedding lengths with an option for custom input
        embedding_lengths = {
            'BERT (768)': 768,
            'DistilBERT (768)': 768,
            'GPT-

