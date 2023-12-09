"""

Script for training and testing the fine-tuned BERT model on the Emotions dataset

"""

from model.emotions_bert import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from bert_recommender import BertRecommender
from spotify_dataset import SpotifyDataset

# Read in data as Pandas dataframes
train_df = pd.read_csv('data/emotions/train.csv')
val_df = pd.read_csv('data/emotions/val.csv')
test_df = pd.read_csv('data/emotions/test.csv')

label_encoder = LabelEncoder()

# Use a label encoder for emotion labels
for df in [train_df, val_df, test_df]:
    df['label'] = label_encoder.fit_transform(df['label'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create Emotion datasets for train, test, and val
train_dataset = EmotionSet(
    texts=train_df['text'].values,
    labels=train_df['label'].values,
    tokenizer=tokenizer,
    max_len=128
)

val_dataset = EmotionSet(
    texts=val_df['text'].values,
    labels=val_df['label'].values,
    tokenizer=tokenizer,
    max_len=128
)

test_dataset = EmotionSet(
    texts=test_df['text'].values,
    labels=test_df['label'].values,
    tokenizer=tokenizer,
    max_len=128
)

# Create dataloaders for each partition
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Create model
model = EmotionBERT(num_labels=len(label_encoder.classes_))

# Set up parameters for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Set number of epochs and train
num_epochs = 3
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_dataloader, optimizer, device)
    print(f'Training loss for epoch {epoch}: {train_loss}')

    val_labels, val_preds = evaluate_model(model, val_dataloader, device)
    val_report = classification_report(val_labels, val_preds)
    print("Validation Report:", val_report)

    test_labels, test_preds = evaluate_model(model, test_dataloader, device)
    test_report = classification_report(test_labels, test_preds)
    print("Test Report:", test_report)


# Create probability distributions for spotify dataset
recommender = BertRecommender(model, tokenizer, max_len=128)

spotify_df = pd.read_csv('data/spotify_millsongdata.csv')

recommender.predict_label_probabilities(spotify_df['text'], device)

spotify = SpotifyDataset('data/spotify_millsongdata.csv')

# Get user input
song_choice = input('Enter a song: ')
neighbors = input('How many similar songs would you like?: ')

# Find and print k most similar songs to the user
song_index = spotify.get_index_from_title(song_choice)

neighbor_indices = recommender.get_recommendations(song_index, int(neighbors))

print()
print('Similar Songs!')
for i, idx in enumerate(neighbor_indices):
    print(f'{i}:', spotify.get_title(idx))