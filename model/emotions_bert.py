"""

A fine-tuned BERT implementation for emotion classification

"""

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import torch
from tqdm import tqdm

class EmotionSet(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        """Initializes all important member variables"""
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.texts)
    
    def __getitem__(self, index):
        """Returns the item at the specified index"""
        text = str(self.texts[index])
        label = self.labels[index]

        inputs = self.tokenizer(text, truncation=True, padding='max_length',
                                max_length=self.max_len, return_tensors='pt')
        
        return {
            'input_ids' : inputs['input_ids'].flatten(),
            'attention_mask' : inputs['attention_mask'].flatten(),
            'labels' : torch.tensor(label, dtype=torch.long)
        }
    
class EmotionBERT(torch.nn.Module):
    def __init__(self, num_labels):
        """Initialized pretrained BERT model for classification"""
        super(EmotionBERT, self).__init__()

        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels):
        """Defines a forward propogation step"""
        outputs = self.bert_model(input_ids, attention_mask, labels)
        return outputs

def train_epoch(model, dataloader, optimizer, device):
    """Train the BERT model for one epoch while calculating loss"""
    model.train()

    total_loss = 0

    for batch in tqdm(dataloader, total=len(dataloader), desc="Training"):
        # Ensure all data is on correct device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Produce output 
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        return total_loss / len(dataloader)
    
def evaluate_model(model, dataloader, device):
    """Generate labels and predictions for evaluation"""

    # Put model into evaluation mode
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
            # Ensure data is on correct device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Generate outputs/logts
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Get predicted labels
            predictions = F.softmax(logits, dim=1).argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Return labels and predictions for further evaluation
    return all_labels, all_predictions