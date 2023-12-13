"""

A song recommendation algorithm based on the Naive Bayes Classifier

"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from model.emotions_bayes import EmotionsNaiveBayes
from emotions_dataset_tfidf import EmotionsDatasetTFIDF
from spotify_dataset import SpotifyDataset

nltk.download('punkt')

class NBEmotionPredictor:
    def __init__(self, model, vectorizer):
        """Initialize the model, vectorizer, and probabilities"""
        self.model = model
        self.vectorizer = vectorizer

        self.probabilities = None

    def preprocess_text(self, text):
        """Remove punctuation and make everything lowercase as is the training data"""
        words = nltk.word_tokenize(text)
        
        words_without_punct = [word for word in words if word.isalnum()]

        text_without_punct = ' '.join(words_without_punct)

        return text_without_punct.lower()

    def predict_label_probabilities(self, new_data):
        """Predict probabilities for all new datapoints"""
        X_new = self.vectorizer.transform(new_data.apply(self.preprocess_text))

        self.probabilities = self.model.model.predict_proba(X_new)
    
    def get_recommendations(self, target_index, k=5):
        """Use cosine similarity to find the k most similar items by label probability distribution"""
        target_distribution = self.probabilities[target_index]

        similarities = cosine_similarity([target_distribution], self.probabilities)[0]

        most_similar_indices = np.argsort(similarities)[-k-1:-1][::-1]

        return most_similar_indices
    

# Load the emotions dataset
# dataset = EmotionsDatasetTFIDF('data/emotions/train.txt',
#                                 'data/emotions/val.txt',
#                                 'data/emotions/test.txt')

# # Create the emotions classifier
# emotion_classifier = EmotionsNaiveBayes(dataset)

# # Train the emotions classifier
# emotion_classifier.train_model()

# # Create a recommendation predictor object
# label_predictor = NBEmotionPredictor(model=emotion_classifier, vectorizer=emotion_classifier.dataset.tfidf_vectorizer)

# # Load spotify data
# spotify = SpotifyDataset('data/spotify_millsongdata.csv')

# # Predict probabilities for spotify songs
# label_predictor.predict_label_probabilities(spotify.df['text'])

# # Get user input
# song_choice = input('Enter a song: ')
# neighbors = input('How many similar songs would you like?: ')

# # Find and print k most similar songs to the user
# song_index = spotify.get_index_from_title(song_choice)

# neighbor_indices = label_predictor.get_recommendations(song_index, int(neighbors))

# print()
# print('Similar Songs!')
# for i, idx in enumerate(neighbor_indices):
#     print(f'{i}:', spotify.get_title(idx))