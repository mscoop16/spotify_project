from sklearn.metrics.pairwise import cosine_similarity
from spotify_dataset import SpotifyDataset
from model.tf_idf import TFIDFGenerator

class SongRecommendationSystem:
    def __init__(self, dataset, tfidf_generator):
        """Create object with dataset and tf-idf generator"""
        self.dataset = dataset
        self.tfidf_generator = tfidf_generator

    def get_recommendations(self, song_index, num_recommendations=5):
        """Get the k most similar songs by tf-idf"""

        tfidf_matrix = self.tfidf_generator.tfidf_matrix

        # Calculate cosine similarity between the given song and all other songs
        cosine_similarities = cosine_similarity(tfidf_matrix[song_index], tfidf_matrix).flatten()

        # Get indices of songs sorted by similarity
        similar_song_indices = cosine_similarities.argsort()[:-1][::-1]

        # Get the top k recommendations
        recommendations = similar_song_indices[:num_recommendations]

        return recommendations

    def print_sample_with_recommendations(self, sample_size=5, num_recommendations=5):
        """Print the k closest songs by tf-idf using random sampling"""
        # Get a random sample of songs
        sample_indices = self.dataset.df.sample(n=sample_size, random_state=16).index

        for song_index in sample_indices:
            song_title = self.dataset.get_title(song_index)

            # Get recommendations for the current song
            recommendations = self.get_recommendations(song_index, num_recommendations)

            print(f"Song: {song_title}")
            print("Recommendations:")
            for i, recommended_index in enumerate(recommendations, start=1):
                recommended_title = self.dataset.get_title(recommended_index)
                print(f"{i}. {recommended_title}")
            print("\n")

if __name__ == "__main__":
    dataset_path = 'data/spotify_millsongdata.csv'
    spotify_dataset = SpotifyDataset(dataset_path)

    tfidf_generator = TFIDFGenerator(spotify_dataset)

    recommendation_system = SongRecommendationSystem(spotify_dataset, tfidf_generator)

    recommendation_system.print_sample_with_recommendations(sample_size=5, num_recommendations=5)
