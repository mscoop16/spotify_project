"""

File for loading in the Emotions for NLP dataset for use with tf-idf features

"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class EmotionsDatasetTFIDF:
    def __init__(self, train_path, val_path, test_path):
        """Set the path for all of the data partitions and the vectorizer"""
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    def load_data(self, path):
        """Load in the data to a Pandas df from the specified path"""
        df = pd.read_csv(path, sep=';', names=['text', 'emotion'])
        return df

    def preprocess_text(self, text):
        """Ensure text is lowered for proper TF-IDF vectorization"""
        return text.lower()

    def prepare_data(self):
        """Translate each file into usable TF-IDF features"""
        # Load the data
        train_df = self.load_data(self.train_path)
        val_df = self.load_data(self.val_path)
        test_df = self.load_data(self.test_path)

        # Preprocess the data
        train_df['text'] = train_df['text'].apply(self.preprocess_text)
        val_df['text'] = val_df['text'].apply(self.preprocess_text)
        test_df['text'] = test_df['text'].apply(self.preprocess_text)

        # Generate TF-IDF features
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(train_df['text'])
        X_val_tfidf = self.tfidf_vectorizer.transform(val_df['text'])
        X_test_tfidf = self.tfidf_vectorizer.transform(test_df['text'])

        return X_train_tfidf, X_val_tfidf, X_test_tfidf, train_df['emotion'], val_df['emotion'], test_df['emotion']
