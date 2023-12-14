# Lyric Emotion Analysis for Mood-Based Music Recommendation
Using an NLP for emotions dataset to train a variety of models on song lyric emotion content prediction. Includes applications for song recommendations based on an input song's emotional content.

## File Descriptions
Data on emotions NLP dataset and Spotify songs available in data folder.
Fine tuned BERT model and Naive Bayes model class for emotion classification available in model folder.
Data preprocessing on emotions NLP dataset contained within emotions_dataset_tfidf.py.
Files for the recommender classes are in bert_recommender.py and naive_bayes_recommender.py.
Training and test of models and running of recommender done in train_and_test_EmotionsNB.py and train_and_test_EmotionBERT.py

## Setup

1. Clone the repository
   ```bash
    git clone https://github.com/mscoop16/spotify_project.git
    ```
2. Setup python virtual environment and install packages
   ```bash
    pip install -r requirements.txt
    ```
3. If attempting to run flask server, switch to frontend branch
4. Make binary into executable and run to start flask server
   ```bash
    chmod +x bin/487run
   ./bin/487run
    ```
## Usage

After setting up, you can use this application at <code class="inline-code" >https://localhost:8000</code>. Enter a song title and the number of recommendations you would like to recieve in the form and press submit.
   
    


Matthew Cooper, Pratik Nadipelli
