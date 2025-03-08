import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, backbone, embedding_dim, projection_dim, num_classes, dropout=0.2):
        """Initialize the projection head wrapper."""
        super().__init__()
        self.backbone = backbone
        self.relu = nn.ReLU()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.fc = nn.Linear(projection_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass through the backbone and projection layers."""
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection(x)
        x = self.fc(x)
        return x