import torch 
import torch.nn as nn
from typing import Any, Dict

from model.transformer import TransformerEncoder
from model.autoencoder import ImgEncoder

class TransformerAdapter(nn.Module):
    def __init__(self, config: Any):
        super().__init__()

        # Target embedding
        layers = [nn.Linear(config.target_dim, config.embed_dim)]
        for _ in range(config.n_embed_layers - 1):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(config.embed_dim, config.embed_dim))
        self.target_embed = nn.Sequential(*layers)

        # Image embedding
        self.img_encoder = ImgEncoder(
            img_dim=config.img_dim,
            kernel_size=config.kernel_size,
            embed_dim=config.embed_dim,
            conv_dims=config.conv_dims,
            fc_dims=config.fc_dims,
        )
        self.img_embed = nn.Sequential(
            nn.Tanh(), 
            nn.Linear(config.embed_dim, config.embed_dim)
        )

        # Attention based encoder
        self.encoder = TransformerEncoder(
            embed_dim=config.embed_dim,
            depth=config.depth, 
            num_heads=config.num_heads, 
            head_dim=config.head_dim, 
            mlp_hidden_dim=config.mlp_hidden_dim, 
            n_features=config.n_features,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            dropout=config.dropout
        )
        self.output = nn.Tanh()

    def forward(self, x: Dict) -> torch.Tensor:
        feat_target = self.target_embed(x['target'])
        feat_img, _ = self.img_encoder(x['img'])
        feat_img = self.img_embed(feat_img)

        embed = torch.stack([feat_target, feat_img], dim=1)
        
        out = self.encoder(embed)
        return self.output(out)
    
    def load_img_encoder(self, path: str=None, device: str=None, require_grad: bool=False) -> None:
        autoencoder = torch.load(path, map_location=device)
        self.img_encoder = autoencoder.encoder
        for param in self.img_encoder.parameters():
            param.requires_grad = require_grad