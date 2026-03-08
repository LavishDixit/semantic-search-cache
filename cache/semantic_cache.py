import numpy as np

class SemanticCache:

    def __init__(self, threshold=0.85):

        self.cache = []
        self.threshold = threshold

        self.hit_count = 0
        self.miss_count = 0


    def cosine_similarity(self, a, b):

        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


    def search(self, query_embedding):

        best_score = 0
        best_entry = None

        for entry in self.cache:

            score = self.cosine_similarity(query_embedding, entry["embedding"])

            if score > best_score:
                best_score = score
                best_entry = entry


        if best_score > self.threshold:

            self.hit_count += 1
            return True, best_entry, best_score


        self.miss_count += 1
        return False, None, best_score


    def add(self, query_embedding, query, result, cluster):

        self.cache.append({
            "embedding": query_embedding,
            "query": query,
            "result": result,
            "cluster": cluster
        })


    def stats(self):

        total = len(self.cache)

        return {
            "total_entries": total,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count/(self.hit_count+self.miss_count) if (self.hit_count+self.miss_count)>0 else 0
        }


    def clear(self):

        self.cache = []

        self.hit_count = 0
        self.miss_count = 0