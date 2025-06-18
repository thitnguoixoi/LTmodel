import torch
import torch.nn as nn

# Define the BiLSTM model with optimizations
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=True, 
                           batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        # Multiply by 2 for bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_length, embedding_dim)
        
        # Pack padded sequence for faster processing
        # This helps with variable length sequences
        packed_output, (hidden, cell) = self.lstm(embedded)
        
        # Use the concatenated hidden state from the last layer
        # Get the last hidden state of both directions
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_cat = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        # Apply dropout
        out = self.dropout(hidden_cat)
        
        # Linear layer
        out = self.fc(out)
        # out shape: (batch_size, num_classes)
        
        return out