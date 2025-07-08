import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, dropout_prob=0.25):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, device=device)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(hidden_dim, device=device)
        self.dropout = nn.Dropout(dropout_prob) 
        self.fc2 = nn.Linear(hidden_dim, output_dim, device=device)

    def forward(self, x):
        x = self.fc1(x).squeeze()
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x).unsqueeze(dim=1)
        return x
