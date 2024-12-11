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

class TextCNN(nn.Module):
    def __init__(self, input_size, num_classes, kernel_sizes, num_filters, dropout=0.5):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, input_size)) for k in kernel_sizes if k <= 1
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # [32, 1, 768]
        
        conv_outputs = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # Conv + ReLU
        pooled_outputs = [torch.max(output, dim=2)[0] for output in conv_outputs]  # Max Pooling
        out = torch.cat(pooled_outputs, dim=1) 
        out = self.dropout(out)
        out = self.fc(out)
        return out

model = torch.load('/Users/habibaalaa/Forum-Discussions-Classification/textcnn.pth',map_location=torch.device('cpu'))

df_test=pd.read_csv('/Users/habibaalaa/Forum-Discussions-Classification/test.csv')

df_test['Discussion'] = df_test['Discussion'].fillna('No Text')
def replace_dates(text):
    date_pattern = r'\b(\d{1,2}-[A-Za-z]{3}|\b[A-Za-z]+ \d{1,2}(\w{2})?)\b'
    return re.sub(date_pattern, '[DATE]', text)

df_test['Discussion'] = df_test['Discussion'].apply(replace_dates)

model_new = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the dataset
embeddings_test = model_new.encode(df_test['Discussion'].tolist(), convert_to_tensor=True).cpu().numpy()

class CustomDataset(Dataset):
    def __init__(self, sentences, sample_ids):
        # Ensure input embeddings are treated as sequences with one channel
        self.sentences = torch.stack([torch.tensor(x, dtype=torch.float32).unsqueeze(0) for x in sentences])
        self.sample_ids = torch.tensor(sample_ids, dtype=torch.long)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.sample_ids[idx]
    
test_dataset = CustomDataset(embeddings_test, df_test['SampleID'])
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

save_csv_path = '/kaggle/working/predictions_textcnn1.csv'

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

test_model(model, test_dataloader, "/Users/habibaalaa/Forum-Discussions-Classification/predictions_TextCNNcsv.csv")



