"""

A song recommendation algorithm based on the BERT Classifier

"""
from model.emotions_bert import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BertRecommender:
    def __init__(self, model, tokenizer, max_len):
        """Initialize the model and probabilities"""
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.probabilities = None
    
    def predict_label_probabilities(self, new_data, device):
        """Predict probabilities for all new datapoints"""
        unlabeled_dataset = EmotionSet(new_data, labels=None, tokenizer=self.tokenizer, max_len=self.max_len)
        unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=False)

        # Put model into evaluation mode
        self.model.eval()
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(unlabeled_dataloader, total=len(unlabeled_dataloader), desc="[Spotify] Predicting Probabilities"):
                # Ensure data is on the correct device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # Generate outputs/logits
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Apply softmax to get probability distribution
                probabilities = F.softmax(logits, dim=1)
                all_probabilities.extend(probabilities.cpu().numpy())

        self.probabilities = all_probabilities
    
    def get_recommendations(self, target_index, k=5):
        """Use cosine similarity to find the k most similar items by label probability distribution"""
        target_distribution = self.probabilities[target_index]

        similarities = cosine_similarity([target_distribution], self.probabilities)[0]

        most_similar_indices = np.argsort(similarities)[-k-1:-1][::-1]

        return most_similar_indices