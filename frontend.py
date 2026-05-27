import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(
    page_title="Semantic Search System",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Semantic Search System")

st.markdown(
    "AI-powered semantic search using FAISS, "
    "Sentence Transformers, fuzzy clustering, "
    "and semantic caching."
)

query = st.text_input(
    "Enter your search query:"
)

if st.button("Search"):

    if query.strip() == "":

        st.warning("Please enter a query")

    else:

        payload = {
            "query": query
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:

            data = response.json()

            st.success("Search completed")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Cache Hit",
                    str(data["cache_hit"])
                )

            with col2:
                st.metric(
                    "Cluster",
                    data["dominant_cluster"]
                )

            st.subheader("Similarity Score")

            st.write(data["similarity_score"])

            st.subheader("Retrieved Result")

            st.write(data["result"])

        else:

            st.error("API request failed")