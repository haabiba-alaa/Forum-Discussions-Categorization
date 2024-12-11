import re
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score
import ast
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import numpy as np

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        
        # Dropout and BatchNorm
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        device = x.device
        x = x.unsqueeze(1)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))  
        
        last_hidden = out[:, -1, :]  
        last_hidden = self.batch_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)

        # Fully connected layer
        output = self.fc(last_hidden)
        return output
    

model = torch.load('/Users/habibaalaa/Forum-Discussions-Classification/bilstm_model_entire.pth',map_location=torch.device('cpu'))

df_test=pd.read_csv('/Users/habibaalaa/Forum-Discussions-Classification/test.csv')

df_test['Discussion'] = df_test['Discussion'].fillna('No Text')
def replace_dates(text):
    date_pattern = r'\b(\d{1,2}-[A-Za-z]{3}|\b[A-Za-z]+ \d{1,2}(\w{2})?)\b'
    return re.sub(date_pattern, '[DATE]', text)

df_test['Discussion'] = df_test['Discussion'].apply(replace_dates)

model_new = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the dataset
embeddings_test = model_new.encode(df_test['Discussion'].tolist(), convert_to_tensor=True).cpu().numpy()

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, sample_ids):
        # Store sentences as a tensor
        self.sentences = torch.stack([torch.tensor(x, dtype=torch.float32) for x in sentences])   # Converting string to list using eval
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.sentences)  # Use self.sentences here

    def __getitem__(self, idx):
        # Return sentence embeddings and the corresponding sample ID
        return self.sentences[idx], self.sample_ids[idx]

test_dataset = CustomDataset(embeddings_test, df_test['SampleID'])
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


def test_model(model, val_dataloader, save_csv_path='predictions.csv', device='cpu'):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    sample_ids = []  # To store sample IDs

    with torch.no_grad():
        for sentences, ids in val_dataloader:  # Extract sentences and IDs from DataLoader
            sentences = sentences.float().to(device)  # Move sentences to GPU (or CPU if needed)
            outputs = model(sentences)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())  # Move predictions back to CPU
            sample_ids.extend(ids.numpy())  # Collect the sample IDs

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame({
        'SampleID': sample_ids,
        'Category': all_preds
    })
    predictions_df.to_csv(save_csv_path, index=False)
    print(f"Predictions saved to {save_csv_path}")

    return predictions_df

test_model(model, test_dataloader, "/Users/habibaalaa/Forum-Discussions-Classification/predictions_BILSTM1.csv")