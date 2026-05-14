import torch
import torch.nn as nn
from .encoder import GCN

class FrozenSingleGraphEncoder(nn.Module):
    def __init__(self, gcn_model, out_dim, dropout=0.1):
        super().__init__()
        self.gcn = gcn_model
        for param in self.gcn.parameters():
            param.requires_grad = False

        # self.adapter = nn.Sequential(
        #     nn.Linear(gcn_model.encoder[-1].out_dim, out_dim),
        #     nn.LayerNorm(out_dim),  # 添加BatchNorm层
        #     nn.ReLU(),
        #     nn.Linear(out_dim, out_dim),
        # )
        gcn_out_dim = gcn_model.encoder[-1].out_dim
        self.adapter = nn.Sequential(
            nn.LayerNorm(gcn_out_dim),
            nn.Linear(gcn_out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )


    def forward(self, A_norm, X):
        with torch.no_grad():
            z = self.gcn.embed(A_norm, X)
        return self.adapter(z)


def load_frozen_encoder(path, in_dim, hid_dims, out_dim):
    model = GCN(in_dim, hid_dims)
    model.load_state_dict(torch.load(path))
    return FrozenSingleGraphEncoder(model, out_dim)
