import faiss
import numpy as np
import os

class VectorStore:

    def __init__(self, dimension):

        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []


    def add(self, embeddings, texts):

        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)

        self.texts = texts


    def search(self, query_embedding, k=5):

        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []

        for i in indices[0]:
            results.append(self.texts[i])

        return results


    def save_index(self, path="storage/faiss.index"):

        os.makedirs("storage", exist_ok=True)

        faiss.write_index(self.index, path)

        print("FAISS index saved")


    def load_index(self, path="storage/faiss.index"):

        self.index = faiss.read_index(path)

        print("FAISS index loaded")