# Semantic Search with Semantic Cache

## Overview

This project implements a lightweight **semantic search system** using
the **20 Newsgroups dataset**.

The system performs semantic retrieval using: - **Sentence Transformer
embeddings** - **FAISS vector database** - **Fuzzy clustering (Gaussian
Mixture Model)** - **Custom semantic cache** to avoid recomputing
similar queries

The system is exposed through a **FastAPI REST API**.

------------------------------------------------------------------------

## Features

-   Semantic search using **Sentence Transformers**
-   **FAISS vector database** for fast similarity search
-   **Fuzzy clustering** using Gaussian Mixture Model
-   Custom **semantic cache implementation**
-   FastAPI service with REST endpoints
-   Cache statistics tracking

------------------------------------------------------------------------

## Architecture

User Query\
↓\
Text Embedding (Sentence Transformer)\
↓\
Semantic Cache Lookup

If Cache Hit → Return Cached Result

If Cache Miss →\
FAISS Vector Search\
↓\
Fuzzy Clustering\
↓\
Store Result in Cache\
↓\
Return Result

------------------------------------------------------------------------

## Tech Stack

-   Python
-   FastAPI
-   Sentence Transformers
-   FAISS
-   Scikit‑learn
-   NumPy
-   Pandas

------------------------------------------------------------------------

## Dataset

The project uses the **20 Newsgroups dataset** which contains
approximately **20,000 news articles across 20 categories**.

For faster development and testing, the **mini_newsgroups subset (\~2000
documents)** was used.

Dataset structure example:

mini_newsgroups/ - alt.atheism - comp.graphics - sci.space - rec.autos -
rec.sport.baseball - talk.politics.misc - talk.religion.misc

------------------------------------------------------------------------

## Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/semantic-search-cache.git

cd semantic-search-cache

Create virtual environment:

python -m venv venv

Activate environment (Windows):

venv\\Scripts\\activate

Install dependencies:

pip install -r requirements.txt

------------------------------------------------------------------------

## Running the API

Start the FastAPI server:

uvicorn main:app --reload

Open browser:

http://127.0.0.1:8000/docs

Swagger UI will appear where you can test the API.

------------------------------------------------------------------------

## API Endpoints

### 1. Query API

POST /query

Request Example:

{ "query": "space shuttle launch" }

Response Example:

{ "query": "space shuttle launch", "cache_hit": false, "matched_query":
null, "similarity_score": 0.0, "result": "...", "dominant_cluster": 3 }

------------------------------------------------------------------------

### 2. Cache Statistics

GET /cache/stats

Example response:

{ "total_entries": 5, "hit_count": 3, "miss_count": 2, "hit_rate": 0.6 }

------------------------------------------------------------------------

### 3. Clear Cache

DELETE /cache

Response:

{ "message": "Cache cleared" }

------------------------------------------------------------------------

## Design Decisions

### Embedding Model

`all-MiniLM-L6-v2` was selected because it provides a strong balance
between **semantic accuracy and computational efficiency**.

### Vector Database

**FAISS** was chosen for fast similarity search in high‑dimensional
vector spaces.

### Clustering

**Gaussian Mixture Model (GMM)** was used to provide **soft
probabilistic cluster assignments**, allowing documents to belong to
multiple semantic topics.

### Semantic Cache

A custom semantic cache was implemented using **cosine similarity
between query embeddings** to detect semantically similar queries and
avoid redundant computation.

------------------------------------------------------------------------

## Example Query Flow

Query: "space shuttle launch"

Embedding Generated\
↓\
Cache Lookup → MISS\
↓\
FAISS Similarity Search\
↓\
Cluster Assignment\
↓\
Result Stored in Cache

Second similar query:

Query: "launch of space shuttle"

Cache Lookup → HIT\
Result returned instantly

------------------------------------------------------------------------

## Future Improvements

-   Persistent FAISS index
-   Persistent semantic cache
-   Docker containerization
-   Query similarity threshold tuning
-   Cluster visualization

------------------------------------------------------------------------

## Author

Lavish Dixit\
AI / ML Student
