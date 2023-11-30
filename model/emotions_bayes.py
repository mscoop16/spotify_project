"""

Implementation of a basic Multinomial Naive Bayes emotion classifier

"""

import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from emotions_dataset_tfidf import EmotionsDatasetTFIDF

class EmotionsNaiveBayes:
    def __init__(self, dataset):
        """Initialize the datatset, model, and label_binarizer"""
        self.dataset = dataset
        self.model = MultinomialNB()

        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.dataset.prepare_data()

        self.lb = LabelBinarizer()
        self.lb.fit(self.y_train)

    def train_model(self):
        """Train our model on the training data"""
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self, set_name):
        """Evaulate the model using f1-score on the specified set"""
        # Make predictions on the set
        X_set = None
        y_set = None
        if set_name == 'train':
            X_set = self.X_train
            y_set = self.y_train
        elif set_name == 'val':
            X_set = self.X_val
            y_set = self.y_val
        else:
            X_set = self.X_test
            y_set = self.y_test

        y_pred = self.model.predict(X_set)

        # Calculate F1-score
        f1 = f1_score(y_set, y_pred, average='weighted')

        print(f"{set_name} F1-Score: {f1}")
        print(f"{set_name} Classification Report:\n", classification_report(y_set, y_pred))

        return f1

    def plot_f1_bar(self):
        """Function for plotting f1-scores of train, validation, and test set"""
        sets = ['train', 'val', 'test']
        f1_scores = []

        for set_name in sets:
            f1_score = self.evaluate_model(set_name)
            f1_scores.append(f1_score)

        plt.bar(sets, f1_scores, color=['blue', 'orange', 'green'])
        plt.ylabel('F1-Score')
        plt.title('Emotions NB: Train vs Validation vs Test F1-Scores')
        plt.show()

    def plot_confusion_matrix(self, set_name):
        """Plot confusion matrix of naive bayes classifier"""
        X_set = None
        y_set = None

        if set_name == 'train':
            X_set, y_set = self.X_train, self.y_train
        elif set_name == 'val':
            X_set, y_set = self.X_val, self.y_val
        else:
            X_set, y_set = self.X_test, self.y_test

        # Get the confusion matrix
        cm = confusion_matrix(y_set, self.model.predict(X_set))

        # Plot the confusion matrix using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.lb.classes_,
                    yticklabels=self.lb.classes_)
        plt.title(f'Navive Bayes Confusion Matrix - {set_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
