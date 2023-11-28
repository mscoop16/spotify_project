import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dataset import SpotifyDataset
from model.tf_idf import TFIDFGenerator

class KMeansClustering:
    def __init__(self, dataset, tfidf_generator):
        """Create dataset and tf_idf objects"""

        self.dataset = dataset
        self.tfidf_generator = tfidf_generator

    def run_kmeans(self, max_clusters=10):
        """Run k-means for k=1 though k=10 clusters"""
        # Retrieve TF-IDF matrix
        tfidf_matrix = self.tfidf_generator.tfidf_matrix

        # Run k-means clustering for different numbers of clusters
        inertias = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=16, n_init='auto')
            kmeans.fit(tfidf_matrix)
            inertias.append(kmeans.inertia_)

        return inertias

    def plot_elbow_method(self, max_clusters=10):
        """Generate an eblow method plot for number of clusters investigation"""
        inertias = self.run_kmeans(max_clusters)

        # Plot the elbow method
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_clusters + 1), inertias, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.show()

if __name__ == "__main__":
    dataset_path = 'data/spotify_millsongdata.csv'
    spotify_dataset = SpotifyDataset(dataset_path)

    tfidf_generator = TFIDFGenerator(spotify_dataset)

    kmeans_clustering = KMeansClustering(spotify_dataset, tfidf_generator)

    kmeans_clustering.plot_elbow_method(max_clusters=10)