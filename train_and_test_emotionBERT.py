"""

Script for training and testing the fine-tuned BERT model on the Emotions dataset

"""

from model.emotions_bert import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    texts=test_df['text'],
    labels=test_df['labels'],
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
optimizer = AdamW(model.parameters(), lr=2e-5)

# Set number of epochs and train
num_epochs = 3
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_dataloader, optimizer, device)

    val_labels, val_preds = evaluate_model(model, val_dataloader, device)
    val_report = classification_report(val_labels, val_preds)