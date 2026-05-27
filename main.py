from fastapi import FastAPI
from api.schemas import QueryRequest

from utils.data_loader import load_dataset
from utils.text_cleaner import clean_text

from models.embedding_model import EmbeddingModel
from vector_db.faiss_store import VectorStore
from models.clustering_model import ClusteringModel

from cache.semantic_cache import SemanticCache

import numpy as np
import os

app = FastAPI(title="Semantic Search API")


print("Loading dataset...")

texts, labels = load_dataset()

cleaned_texts = [clean_text(t) for t in texts]


print("Loading embedding model...")

embedding_model = EmbeddingModel()


# ----------------------------------------
# LOAD OR GENERATE EMBEDDINGS
# ----------------------------------------

if os.path.exists("storage/embeddings.npy"):

    print("Loading saved embeddings...")

    embeddings = np.load("storage/embeddings.npy")

else:

    print("Generating embeddings...")

    embeddings = embedding_model.encode(cleaned_texts)

    os.makedirs("storage", exist_ok=True)

    np.save("storage/embeddings.npy", embeddings)

    print("Embeddings saved")


# ----------------------------------------
# LOAD OR CREATE FAISS INDEX
# ----------------------------------------

dimension = embeddings.shape[1]

vector_store = VectorStore(dimension)

if os.path.exists("storage/faiss.index"):

    print("Loading FAISS index...")

    vector_store.load_index()

    vector_store.texts = cleaned_texts

else:

    print("Creating FAISS index...")

    vector_store.add(embeddings, cleaned_texts)

    vector_store.save_index()

    print("FAISS index saved")


# ----------------------------------------
# TRAIN CLUSTERING MODEL
# ----------------------------------------

print("Training clustering model...")

cluster_model = ClusteringModel()

cluster_model.fit(embeddings)


# ----------------------------------------
# INITIALIZE SEMANTIC CACHE
# ----------------------------------------

print("Initializing semantic cache...")

semantic_cache = SemanticCache()


# ----------------------------------------
# API ENDPOINTS
# ----------------------------------------

@app.get("/")
def home():

    return {"message": "Semantic Search API Running"}


@app.post("/query")
def query_api(request: QueryRequest):

    query = request.query

    query_embedding = embedding_model.encode([query])[0]

    hit, entry, score = semantic_cache.search(query_embedding)

    if hit:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(score),
            "result": entry["result"][:500],
            "dominant_cluster": int(entry["cluster"])
        }

    results = vector_store.search(query_embedding)

    cluster_probs = cluster_model.get_cluster_distribution(query_embedding)

    dominant_cluster = cluster_probs.argmax()

    semantic_cache.add(
        query_embedding,
        query,
        results[0],
        dominant_cluster
    )

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": float(score),
        "result": results[0][:500],
        "dominant_cluster": int(dominant_cluster)
    }


@app.get("/cache/stats")
def cache_stats():

    return semantic_cache.stats()


@app.delete("/cache")
def clear_cache():

    semantic_cache.clear()

    return {"message": "Cache cleared"}