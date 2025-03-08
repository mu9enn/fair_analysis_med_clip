import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, backbone, embedding_dim, projection_dim, num_classes, dropout=0.2):
        """Initialize the MLP head wrapper."""
        super().__init__()
        self.backbone = backbone
        self.batch_norm = nn.BatchNorm1d(projection_dim)
        self.relu = nn.ReLU()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.fc = nn.Linear(projection_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass through the backbone and MLP layers."""
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.fc(x)
        return x