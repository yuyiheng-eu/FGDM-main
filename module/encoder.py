# coding: utf-8
import torch.nn as nn
from torch import Tensor

from .helpers import freeze_params
from .transformer_layers import TransformerEncoderLayer, PositionalEncoding

class Encoder(nn.Module):

    def __init__(self,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 **kwargs):

        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                    num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self._output_size = hidden_size

        if freeze:
            freeze_params(self)

    def forward(self,
                embed_src: Tensor,
                src_length: Tensor,
                mask: Tensor):

        x = embed_src

        # Add position encoding to word embeddings
        x = self.pe(x)
        # Add Dropout
        x = self.emb_dropout(x)

        # Apply each layer to the input
        for layer in self.layers:
            x = layer(x, mask)

        return self.layer_norm(x)

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].src_src_att.num_heads)