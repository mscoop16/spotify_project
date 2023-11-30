import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from model.emotions_bayes import EmotionsNaiveBayes
from emotions_dataset_tfidf import EmotionsDatasetTFIDF
from spotify_dataset import SpotifyDataset

nltk.download('punkt')

class NBLabelPredictor:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

        self.probabilities = None

    def preprocess_text(self, text):
        words = nltk.word_tokenize(text)
        
        words_without_punct = [word for word in words if word.isalnum()]

        text_without_punct = ' '.join(words_without_punct)

        return text_without_punct.lower()

    def predict_label_probabilities(self, new_data):
        X_new = self.vectorizer.transform(new_data.apply(self.preprocess_text))

        self.probabilities = self.model.model.predict_proba(X_new)
    
    def get_recommendations(self, target_index, k=5):
        target_distribution = self.probabilities[target_index]

        similarities = cosine_similarity([target_distribution], self.probabilities)[0]

        most_similar_indices = np.argsort(similarities)[-k-1:-1][::-1]

        return most_similar_indices
    
dataset = EmotionsDatasetTFIDF('data/emotions/train.txt',
                                'data/emotions/val.txt',
                                'data/emotions/test.txt')

emotion_classifier = EmotionsNaiveBayes(dataset)

emotion_classifier.train_model()

# Create an instance of the NBLabelPredictor class
label_predictor = NBLabelPredictor(model=emotion_classifier, vectorizer=emotion_classifier.dataset.tfidf_vectorizer)

# Load the new dataset

spotify = SpotifyDataset('data/spotify_millsongdata.csv')

# Use the class method to predict label probabilities
label_predictor.predict_label_probabilities(spotify.df['text'])

song_choice = input('Enter a song: ')
neighbors = input('How many similar songs would you like?: ')

song_index = spotify.get_index_from_title(song_choice)

neighbor_indices = label_predictor.get_recommendations(song_index, int(neighbors))

print()
print('Similar Songs!')
for i, idx in enumerate(neighbor_indices):
    print(f'{i}:', spotify.get_title(idx))