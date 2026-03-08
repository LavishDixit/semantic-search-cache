from sklearn.mixture import GaussianMixture

class ClusteringModel:

    def __init__(self, n_clusters=15):

        self.model = GaussianMixture(n_components=n_clusters)

    def fit(self, embeddings):

        self.model.fit(embeddings)

    def get_cluster_distribution(self, embedding):

        probs = self.model.predict_proba([embedding])[0]

        return probs