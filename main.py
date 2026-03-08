from fastapi import FastAPI
from api.schemas import QueryRequest

from utils.data_loader import load_dataset
from utils.text_cleaner import clean_text

from models.embedding_model import EmbeddingModel
from vector_db.faiss_store import VectorStore
from models.clustering_model import ClusteringModel

from cache.semantic_cache import SemanticCache

app = FastAPI(title="Semantic Search API")


print("Loading dataset...")

texts, labels = load_dataset()

cleaned_texts = [clean_text(t) for t in texts]


print("Generating embeddings...")

embedding_model = EmbeddingModel()

embeddings = embedding_model.encode(cleaned_texts)


print("Creating FAISS index...")

dimension = embeddings.shape[1]

vector_store = VectorStore(dimension)

vector_store.add(embeddings, cleaned_texts)


print("Training clustering model...")

cluster_model = ClusteringModel()

cluster_model.fit(embeddings)


print("Initializing semantic cache...")

semantic_cache = SemanticCache()


# -----------------------------
# API ENDPOINTS
# -----------------------------


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

    semantic_cache.add(query_embedding, query, results[0], dominant_cluster)

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