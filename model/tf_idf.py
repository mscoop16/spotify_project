"""

A baseline model for comparison using TF-IDF vectorization

"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from spotify_dataset import SpotifyDataset

class TFIDFGenerator:
    def __init__(self, dataset):
        """Initialize TFIDF vectorizer and dataset"""
        
        self.dataset = dataset

        self.vectorizer = TfidfVectorizer()

        self.tfidf_matrix = self.vectorizer.fit_transform(self.dataset.df['text'].fillna(''))
    
    def get_tfidf_vector(self, index):
        """Get the TF-IDF vector for a specific song"""

        return self.tfidf_matrix[index]