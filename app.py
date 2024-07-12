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

        # Display selected columns for debugging
        st.write(f"Selected URL column: {url_column}")
        st.write(f"Selected Embeddings column: {embeddings_column}")

        # Check if the selected columns are in the dataframe
        if url_column not in df.columns or embeddings_column not in df.columns:
            st.error("Selected columns are not in the dataframe.")
        else:
            # Convert embeddings from string to numpy arrays
            st.write("Converting embeddings to numpy arrays...")
            df[embeddings_column] = df[embeddings_column].apply(lambda x: np.array([float(i) for i in str(x).split(',')]))

            # Display the first few rows to ensure the conversion is correct
            st.write("Converted DataFrame:")
            st.write(df[[url_column, embeddings_column]].head())

            # Filter out rows with embeddings that don't have the expected shape of (1536,)
            expected_embedding_length = 1536
            st.write(f"Filtering embeddings with length {expected_embedding_length}...")
            filtered_df = df[df[embeddings_column].apply(lambda x: len(x) == expected_embedding_length)]

            # Display the filtered DataFrame for debugging
            st.write("Filtered DataFrame:")
            st.write(filtered_df[[url_column, embeddings_column]].head())

            # Function to find top N related pages for each URL based on cosine similarity
            def find_related_pages(df, top_n=3):
                related_pages = {}
                embeddings = np.stack(df[embeddings_column].values)
                cosine_similarities = cosine_similarity(embeddings)

                for idx, url in enumerate(df[url_column]):
                    similar_indices = cosine_similarities[idx].argsort()[-(top_n+1):-1][::-1]  # Top N indices, excluding itself
                    related_urls = df.iloc[similar_indices][url_column].values.tolist()
                    related_pages[url] = related_urls

                return related_pages

            # Get the top N related pages from user input
            top_n = st.number_input('Number of related pages to find', min_value=1, max_value=10, value=5)

            # Find related pages
            st.write("Finding related pages...")
            related_pages = find_related_pages(filtered_df, top_n=top_n)

            # Create a graph
            G = nx.Graph()

            # Add nodes and edges to the graph
            st.write("Creating graph...")
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

            st.plotly_chart(fig)
    except KeyError as e:
        st.error(f"Column not found: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

