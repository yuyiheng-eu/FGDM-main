import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from einops import rearrange
from .utils.tgcn import ConvTemporalGraphical, ConvTemporalGraphicalv2
from .utils.graph import Graph
import math


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class asgcn_unit(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 layout='whole',
                 strategy='spatial',
                 ops= 'mul',
                 neighbor=3,
                 residual=True,
                 use_D=True,
                 use_A=True,
                 ):
        super().__init__()

       
        self.ops = ops
        self.residual = residual
        self.use_A = use_A
        self.use_D = use_D
        self.neighbor = neighbor
        self.groups = in_channels
        self.graph = Graph(layout, strategy)
        if self.use_A:
            self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.B = nn.Parameter(torch.from_numpy(self.graph.A.astype(np.float32)))
        kernel_size = self.B.size(0)

        self.Separation = nn.Conv3d(1, kernel_size, 1)

        self.gcn = ConvTemporalGraphicalv2(in_channels, in_channels,
                                         kernel_size, self.groups)

        self.gloss_projector = nn.Linear(512, in_channels)
        self.adjacency_generator = nn.Linear(512, self.graph.num_node**2)

        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.neighbor_linear = nn.Linear(self.neighbor * 2 + 1, kernel_size)
        nn.init.constant_(self.neighbor_linear.weight, 0)
        nn.init.constant_(self.neighbor_linear.bias, 0)

        # Initialize Separation layer weights to zero
        nn.init.constant_(self.Separation.weight, 0)
        nn.init.constant_(self.Separation.bias, 0)

        if self.residual:
            if in_channels != out_channels:
                self.residual_down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.residual_down = lambda x: x

        self.output_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        bn_init(self.bn, 1e-6)


    def forward(self, x, condition, mask):
        ###### Skeletal Topology #######
        if self.use_A:
            adaptive_adjacency = self.A.to(x.device) + self.B
        else:
            adaptive_adjacency = self.B

        ###### Semantic Mask Generation #######
        gloss_feat = self.gloss_projector(condition)
        adj_map = self.adjacency_generator(condition)
        adj_map = rearrange(adj_map, 'n t (v w) -> n t v w', v=self.graph.num_node)

        # Compute attention scores between gloss features and pose features
        scores = torch.einsum('ntc, ncTv -> ntTv', gloss_feat, x) / math.sqrt(gloss_feat.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(~mask.squeeze(1)[..., None, None], float('-inf'))
        scores = F.softmax(scores, dim=-3)

        # Generate semantic mask M
        semantic_mask = torch.einsum('ntTv, ntvw -> nTvw', scores, adj_map)
        semantic_mask = self.Separation(semantic_mask[:, None, ...])
        semantic_mask = semantic_mask.permute(0, 2, 1, 3, 4)


        ###### Contextual Correlation Aggregation #######
        if self.use_D:
            query = self.query_conv(x)
            key = self.key_conv(x)
            key_unfold = F.pad(key, (0, 0, self.neighbor, self.neighbor), mode='constant', value=0).unfold(2, self.neighbor * 2 + 1, 1)
            correlation = F.sigmoid(torch.einsum('nctv, nctwk -> ntvwk', query, key_unfold) / math.sqrt(query.shape[1]))
            dynamic_adjacency = self.neighbor_linear(correlation).permute(0, 1, 4, 2, 3)
        else:
            dynamic_adjacency = torch.zeros_like(adaptive_adjacency[None, None, ...], device=adaptive_adjacency.device)

        ###### Adaptive Graph Convolution #######
        if self.ops == 'mul':
            adaptive_adjacency = (adaptive_adjacency[None, None, ...] + dynamic_adjacency) * F.sigmoid(semantic_mask) * 2
        else:
            adaptive_adjacency = adaptive_adjacency[None, None, ...] + semantic_mask + dynamic_adjacency

        y, _ = self.gcn(x, adaptive_adjacency)
        y = self.output_conv(y)
        y = self.bn(y)
        y = self.dropout(y)
        if self.residual:
            y += self.residual_down(x)
        y = self.relu(y)

        return y




class st_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 Temp_kernel_size=9,
                 stride=1,
                 dropout=0,
                 layout='whole',
                 strategy='spatial',
                 use_A='True',
                 residual=True):
        super().__init__()

        
        self.use_A = use_A
        self.graph = Graph(layout,strategy)
        if self.use_A:
            self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.B = nn.Parameter(torch.from_numpy(self.graph.A.astype(np.float32)))
        kernel_size = (Temp_kernel_size, self.B.size(0))
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        res = self.residual(x)
        if self.use_A:
            self.A = self.A.to(x.device)
            A = self.A + self.B 
        else:
            A = self.B
        x, _ = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x)



class F_Stage(nn.Module):

    def __init__(self,
                 in_channels,
                 Temp_kernel_size=9,
                 stride=1,
                 dropout=0,
                 layout='whole',
                 strategy='spatial',
                 use_A='True',
                 residual=True):
        super().__init__()

        self.gcn = asgcn_unit(
            in_channels=in_channels,
            out_channels=in_channels,
            layout=layout,
            strategy=strategy,
            use_A=use_A,
            residual=residual
        )
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                (Temp_kernel_size, 1),
                (stride, 1),
                ((Temp_kernel_size - 1) // 2, 0),
            ),
            nn.BatchNorm2d(in_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        
        self.residual = lambda x: x

        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, condition, mask):

        res = self.residual(x)

        x = self.gcn(x, condition, mask)
        # return x

        x = self.tcn(x) + res
      

        return self.relu(x)


    