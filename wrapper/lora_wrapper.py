import torch
import torch.nn as nn
import math

class LoRALayer_vit(nn.Module):
    def __init__(self, original_layer, rank=16, alpha=1):
        """Initialize LoRA layer for ViT attention."""
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        if hasattr(original_layer, 'in_proj_weight'):
            in_dim = original_layer.in_proj_weight.shape[1]
            out_dim = original_layer.in_proj_weight.shape[0]
        else:
            in_dim = original_layer.weight.shape[1]
            out_dim = original_layer.weight.shape[0]
        self.rank_down = nn.Parameter(torch.zeros(rank, in_dim))
        self.rank_up = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.rank_down, a=math.sqrt(5))
        nn.init.zeros_(self.rank_up)

    def forward(self, query, key, value, **kwargs):
        """Forward pass with LoRA adjustment for ViT."""
        low_rank_matrix = self.rank_up @ self.rank_down
        if hasattr(self.original_layer, 'in_proj_weight'):
            adjusted_weight = self.original_layer.in_proj_weight + self.alpha * low_rank_matrix
            self.original_layer.in_proj_weight.data = adjusted_weight
        else:
            adjusted_weight = self.original_layer.weight + self.alpha * low_rank_matrix
            self.original_layer.weight.data = adjusted_weight
        return self.original_layer(query, key, value, **kwargs)

class LoRAConv2d(nn.Module):
    def __init__(self, conv_layer, r=4):
        """Initialize LoRA layer for ResNet convolutions."""
        super().__init__()
        self.conv_layer = conv_layer
        self.r = r
        in_channels = conv_layer.in_channels
        out_channels = conv_layer.out_channels
        kernel_size = conv_layer.kernel_size
        self.A = nn.Parameter(torch.randn(out_channels, r, 1, 1) * 0.01)
        self.B = nn.Parameter(torch.randn(r, in_channels, kernel_size[0], kernel_size[1]) * 0.01)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        """Forward pass with LoRA adjustment for convolutions."""
        low_rank_weight = torch.matmul(self.A.view(self.A.size(0), -1), self.B.view(self.B.size(0), -1))
        low_rank_weight = low_rank_weight.view(self.A.size(0), self.B.size(1), self.B.size(2), self.B.size(3))
        original_weight = self.conv_layer.weight + low_rank_weight
        return nn.functional.conv2d(x, original_weight, self.conv_layer.bias,
                                    stride=self.conv_layer.stride, padding=self.conv_layer.padding)

class CLIPWithLoRA(nn.Module):
    def __init__(self, backbone, is_vit, embedding_dim, projection_dim, num_classes, dropout=0.2):
        """Initialize CLIP with LoRA adaptations."""
        super().__init__()
        self.backbone = backbone
        self.is_vit = is_vit
        self.lora_layers = nn.ModuleList()

        if self.is_vit:
            # Apply LoRA to ViT attention layers (assuming 12 layers)
            self.lora_layers = nn.ModuleList([
                LoRALayer_vit(self.backbone.transformer.resblocks[i].attn)
                for i in range(12)
            ])
            for i, layer in enumerate(self.lora_layers):
                self.backbone.transformer.resblocks[i].attn = layer
        else:
            # Apply LoRA to ResNet conv layers
            for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
                for block in getattr(self.backbone, layer_name):
                    block.conv1 = LoRAConv2d(block.conv1)
                    block.conv2 = LoRAConv2d(block.conv2)
                    block.conv3 = LoRAConv2d(block.conv3)
                    self.lora_layers.extend([block.conv1, block.conv2, block.conv3])

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.fc = nn.Linear(projection_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass through the backbone and LoRA-adjusted layers."""
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection(x)
        x = self.fc(x)
        return x