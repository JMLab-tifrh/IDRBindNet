import os
import sys
import torch
import torch.nn.functional as F 
from torch_geometric.nn import TransformerConv, global_mean_pool, BatchNorm
from torch.nn import BatchNorm1d

class GTR(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim=128, heads=8, dropout=0.1):
        super().__init__()
        
        self.heads = heads
        self.dropout = dropout

        # ---------- Edge feature encoder ----------
        # You can change the size (64) if needed.
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, hidden_dim)
        )

        # ---------- TransformerConv layers ----------
        # Layer 1
        self.conv1 = TransformerConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=hidden_dim  # IMPORTANT
        )
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * heads)

        # Layer 2
        self.conv2 = TransformerConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=hidden_dim
        )
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim * heads)

        # Layer 3
        self.conv3 = TransformerConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=hidden_dim
        )
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)

        # ---------- Regression head ----------
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Encode edges for all layers
        e = self.edge_encoder(edge_attr)  # [E, hidden_dim]

        # Layer 1
        x = self.conv1(x, edge_index, e)
        x = self.bn1(x)
        x = F.elu(x)

        # Layer 2
        x = self.conv2(x, edge_index, e)
        x = self.bn2(x)
        x = F.elu(x)

        # Layer 3
        x = self.conv3(x, edge_index, e)
        x = self.bn3(x)
        x = F.elu(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        return self.regressor(x).squeeze(1)
