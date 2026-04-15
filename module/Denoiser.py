# coding: utf-8
import torch
import torch.nn as nn
import math
from torch import Tensor

from einops import rearrange
from .helpers import freeze_params, subsequent_mask
from .transformer_layers import PositionalEncoding, TransformerDecoderLayer
from .gcn import F_Stage
from .ID import ID


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Denoiser(nn.Module):

    def __init__(self,
                 num_focal_layers: int = 3,
                 num_general_layers: int = 2,
                 num_heads: int = 4,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 trg_size: int = 150,
                 decoder_trg_trg_: bool = True,
                 gcn_hidden_size: int = 16,
                 **kwargs):
        super().__init__()

        self.in_feature_size = trg_size
        self.out_feature_size = trg_size
        self.joints = int(trg_size // 3)
        self.num_focal_layers = num_focal_layers
        self.num_general_layers = num_general_layers

        # Focal Stage: ASGCN layers for joint-level modeling
        self.focal_embed = nn.Conv2d(7, gcn_hidden_size, 1)  # 7 = 3D joints + 4D iconicity
        self.Focal = nn.ModuleList([
            F_Stage(gcn_hidden_size) for _ in range(self.num_focal_layers)
        ])

        # Focal → General projection
        self.f2g_embed = nn.Linear(gcn_hidden_size * self.joints, hidden_size)

        # General Stage: Transformer layers for global coherence
        self.pe = PositionalEncoding(hidden_size, mask_count=True)
        self.pos_drop = nn.Dropout(p=emb_dropout)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.General = nn.ModuleList([
            TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout, decoder_trg_trg=decoder_trg_trg_
            ) for _ in range(self.num_general_layers)
        ])

        # Intermediate projection between General stages
        self.layer_norm_mid = nn.LayerNorm(hidden_size, eps=1e-6)
        self.output_layer_mid = nn.Linear(hidden_size, hidden_size, bias=False)

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        # Output projection
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, trg_size, bias=False)
        )

        if freeze:
            freeze_params(self)

    def forward(self,
                t,
                trg_embed: Tensor = None,
                encoder_output: Tensor = None,
                src_mask: Tensor = None,
                trg_mask: Tensor = None,
                **kwargs):

        assert trg_mask is not None, "trg_mask required for Transformer"

        # Timestep conditioning
        time_embed = self.time_mlp(t)[:, None, :].repeat(1, encoder_output.shape[1], 1)
        condition = self.pos_drop(encoder_output + time_embed)

        sub_mask = subsequent_mask(trg_embed.size(1)).type_as(trg_mask)
        padding_mask = trg_mask

        # ===== Focal Stage =====
        trg_embed = ID(trg_embed)
        trg_f = self.focal_embed(rearrange(trg_embed, 'n t (v c) -> n c t v', v=50))

        for stage in self.Focal:
            trg_f = stage(trg_f, condition, mask=src_mask)

        # Focal → General transition
        trg_g = self.emb_dropout(self.pe(self.f2g_embed(
            rearrange(trg_f, 'n c t v -> n t (v c)'))))

        # ===== General Stage =====
        for idx, stage in enumerate(self.General):
            trg_g, _ = stage(x=trg_g, memory=condition,
                             src_mask=src_mask, trg_mask=sub_mask,
                             padding_mask=padding_mask)
            if idx < 1:
                trg_g = self.output_layer_mid(self.layer_norm_mid(trg_g))

        output = self.output_layer(trg_g)
        return output, trg_g

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].trg_trg_att.num_heads)


if __name__ == "__main__":
    model = Denoiser()
    shape_trg = (64, 1, 129, 129)
    shape_src = (64, 1, 9)
    prob = 0.3
    src_mask = torch.bernoulli(torch.ones(shape_src) * (1 - prob)).bool()
    trg_mask = torch.bernoulli(torch.ones(shape_trg) * (1 - prob)).bool()
    t = torch.ones(64) * 999
    print(model(t, torch.randn(64, 129, 150), torch.randn(64, 9, 512), src_mask, trg_mask).shape)
