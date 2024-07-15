# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objs as go
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit app
st.title("URL Relationships App")

# List of models and their embedding lengths with descriptions (for user information)
models = {
    "text-embedding-ada-002": ("1536 dimensions - OpenAI: General purpose embeddings, good balance of performance and cost.", 1536),
    "text-similarity-babbage-001": ("2048 dimensions - OpenAI: Suitable for tasks requiring text similarity measures.", 2048),
    "text-similarity-curie-001": ("4096 dimensions - OpenAI: More powerful text similarity model for complex tasks.", 4096),
    "text-similarity-davinci-001": ("4096 dimensions - OpenAI: Most powerful model for the most complex text similarity tasks.", 4096),
    "BERT-base": ("768 dimensions - BERT: General-purpose embeddings for various NLP tasks.", 768),
    "BERT-large": ("1024 dimensions - BERT: More powerful model with larger embeddings.", 1024),
    "GPT-3": ("12288 dimensions - GPT-3: Advanced model with large embedding size for complex tasks.", 12288)
}

# Model selection (for informational purposes)
model_description, expected_length = st.selectbox("Select the embedding model (for info only)", list(models.values()), format_func=lambda x: x[0])

# Extract model key from the selected description
model = [key for key, value in models.items() if value == (model_description, expected_length)][0]

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read file
    df = pd.read_csv(uploaded_file)
    st.write("File successfully uploaded!")

    # Display columns and let user map them
    url_column = st.selectbox("Select URL column", df.columns)
    embedding_column = st.selectbox("Select Embeddings column", list(df.columns))

    # Ensure the 'URL' column is mapped
    if 'URL' not in df.columns:
        df['URL'] = df[url_column]

    # Convert embeddings from string to numpy array
    df['Embeddings'] = df[embedding_column].apply(lambda x: np.array([float(i) for i in str(x).split(',')]))

    # Filter correct shape
    filtered_df = df[df['Embeddings'].apply(lambda x: len(x) == expected_length)]

    # Display filtered dataframe
    st.write("Filtered DataFrame", filtered_df.head())

    # Function to find top N related pages for each URL based on cosine similarity
    def find_related_pages(df, top_n=5):
        related_pages = {}
        embeddings = np.stack(df['Embeddings'].values)
        cosine_similarities = cosine_similarity(embeddings)

        for idx, url in enumerate(df['URL']):
            similar_indices = cosine_similarities[idx].argsort()[-(top_n+1):-1][::-1]
            related_urls = df.iloc[similar_indices]['URL'].values.tolist()
            related_pages[url] = related_urls

        return related_pages

    # Find related pages
    related_pages = find_related_pages(filtered_df, top_n=5)

    # Convert the result to a DataFrame for easier inspection
    related_pages_df = pd.DataFrame(list(related_pages.items()), columns=['URL', 'Related URLs'])

    # Save the result to a CSV file
    output_file_name = 'related_pages_filtered.csv'
    related_pages_df.to_csv(output_file_name, index=False)
    st.write(f"Results saved to {output_file_name}")

    # Provide download button
    st.download_button(
        label="Download data as CSV",
        data=related_pages_df.to_csv(index=False).encode('utf-8'),
        file_name=output_file_name,
        mime='text/csv',
    )

    # Create a graph
    G = nx.Graph()

    # Add nodes and edges to the graph
    for url, related_urls in related_pages.items():
        G.add_node(url)
        for related_url in related_urls:
            G.add_node(related_url)
            G.add_edge(url, related_url)

    # Create the Plotly graph
    pos = nx.spring_layout(G, k=0.1)  # positions for all nodes

    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_color = []

    for node, adjacencies in enumerate(G.adjacency()):
        node_color.append(len(adjacencies[1]))
        node_info = adjacencies[0]
        node_text.append(node_info)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=node_color,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='<br>URL Relationships',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper")],
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False))
                    )

    # Display the graph only once
    st.plotly_chart(fig)

