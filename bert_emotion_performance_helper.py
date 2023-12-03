"""

A helper file for plotting the BERT output

"""

import pandas as pd
import matplotlib.pyplot as plt

training_data = {
    'epoch': [0, 1, 2],
    'training_loss': [0.4815612161522731, 0.12949383133882655, 0.0994928853737656]
}
training_df = pd.DataFrame(training_data)

validation_data = {
    'epoch': [0, 1, 2],
    'accuracy': [0.93, 0.94, 0.94],
    'macro_f1_score': [0.90, 0.91, 0.91],
    'weighted_f1_score': [0.93, 0.93, 0.94]
}
validation_df = pd.DataFrame(validation_data)

test_data = {
    'epoch': [0, 1, 2],
    'accuracy': [0.93, 0.93, 0.93],
    'macro_f1_score': [0.89, 0.88, 0.88],
    'weighted_f1_score': [0.93, 0.93, 0.93]
}

test_df = pd.DataFrame(test_data)

plt.figure(figsize=(10, 6))

plt.plot(validation_df['epoch'], validation_df['accuracy'], label='Validation Accuracy', marker='o', color='blue')
plt.plot(test_df['epoch'], test_df['accuracy'], label='Test Accuracy', marker='o', color='orange')

plt.plot(validation_df['epoch'], validation_df['macro_f1_score'], label='Validation Macro F1 Score', marker='o', color='green')
plt.plot(test_df['epoch'], test_df['macro_f1_score'], label='Test Macro F1 Score', marker='o', color='red')

plt.plot(validation_df['epoch'], validation_df['weighted_f1_score'], label='Validation Weighted F1 Score', marker='o', color='purple')
plt.plot(test_df['epoch'], test_df['weighted_f1_score'], label='Test Weighted F1 Score', marker='o', color='brown')

plt.title('Fine-Tuned BERT: Evaluation Metrics Over Epochs')
plt.xlabel('Epoch')
plt.xticks([0, 1, 2])
plt.ylabel('Percentage')
plt.legend(loc='lower right', bbox_to_anchor=(1, 0.18))

plt.show()

plt.figure(figsize=(8, 6))
plt.bar(training_df['epoch'], training_df['training_loss'], color='skyblue')
plt.title('Fine-Tuned BERT: Training Loss Over Epochs')
plt.xticks([0, 1, 2])
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.show()