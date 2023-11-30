"""

Script for using our Naive Bayes based Emotions classifier

"""

from model.emotions_bayes import EmotionsNaiveBayes
from emotions_dataset_tfidf import EmotionsDatasetTFIDF

dataset = EmotionsDatasetTFIDF('data/emotions/train.txt',
                                'data/emotions/val.txt',
                                'data/emotions/test.txt')

emotion_classifier = EmotionsNaiveBayes(dataset)

emotion_classifier.train_model()
emotion_classifier.plot_f1_bar()

emotion_classifier.plot_confusion_matrix('val')