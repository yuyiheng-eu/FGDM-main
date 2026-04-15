import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_layers import PositionalEncoding, TransformerDecoderLayer
import math
from helpers import freeze_params, subsequent_mask
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
def modulate(x, shift, scale, mod):
    if mod=="1":
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    else:
        return x * (1 + scale) + shift
class CrossAttBlock(nn.Module):
   
    def __init__(
                self, 
                hidden_size, 
                num_heads,  
                ff_size,
                dropout, 
                decoder_trg_trg_,
                mlp_ratio=1/16,
                mod = '1'
                ):
        super().__init__()
        self.mod = mod
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.crossattn = TransformerDecoderLayer(
            size=hidden_size, ff_size=ff_size, num_heads=num_heads,
            dropout=dropout, decoder_trg_trg=decoder_trg_trg_)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, condition, src_mask, trg_mask, padding_mask):
        sub_mask = subsequent_mask(
            x.size(1)).type_as(trg_mask)
        if self.mod == "1":
            condition_ = condition.mean(dim=1)#n t c -> n c
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition_).chunk(6, dim=1)
            print(modulate(self.norm1(x), shift_msa, scale_msa,self.mod).shape)
            x = x + gate_msa.unsqueeze(1) * self.crossattn(x=modulate(self.norm1(x), shift_msa, scale_msa,self.mod), memory=condition,
                                src_mask=src_mask, trg_mask=sub_mask, padding_mask=padding_mask)[0]
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp,self.mod))
        if self.mod == "2":
            corr = torch.einsum('nTc, ntc -> nTt', x, condition)/math.sqrt(condition.size(1))
            condition_ = torch.einsum('nTt, ntc -> nTc', F.sigmoid(corr)-0.5, condition)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition_).chunk(6, dim=-1)
            x = x + gate_msa * self.crossattn(x=modulate(self.norm1(x), shift_msa, scale_msa, self.mod), memory=condition,
                                src_mask=src_mask, trg_mask=sub_mask, padding_mask=padding_mask)[0]
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp,self.mod))
     
        return x
if __name__ == "__main__":
    model = CrossAttBlock( 
                hidden_size=512, 
                num_heads=8, 
                mlp_ratio=1/16, 
                ff_size=2048,
                dropout=0.1, 
                decoder_trg_trg_=True,
                mod = '2')
    shape_trg = (64,1,129,129)
    shape_src = (64,1,9)
    prob = 0.3  # 30%的元素会被掩盖（设置为0）
    src_mask = torch.bernoulli(torch.ones(shape_src) * (1 - prob)).bool()
    trg_mask = torch.bernoulli(torch.ones(shape_trg) * (1 - prob)).bool()
    print(model(torch.randn(64,129,512),torch.randn(64,9,512),src_mask,trg_mask,trg_mask).shape)
