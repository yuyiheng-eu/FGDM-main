import torch
import torch.nn as nn
import torch.nn.functional as F
from .criterions import SeqKD
from .tconv import TemporalConv
from .BiLSTM import BiLSTMLayer


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        return torch.matmul(x, F.normalize(self.weight, dim=0))


class SCGModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super(SCGModel, self).__init__()

        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = cfg["loss_weights"]

        input_size = cfg["input_size"]
        hidden_size = cfg["hidden_size"]
        self.adapter = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, input_size)
        )
        self.conv1d = TemporalConv(
            input_size=512,
            hidden_size=hidden_size,
            conv_type=cfg["conv_type"],
            use_bn=cfg["use_bn"],
            num_classes=self.num_classes
        )
        self.temporal_model = BiLSTMLayer(
            rnn_type='LSTM',
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True
        )
        self.classifier = NormLinear(hidden_size, self.num_classes)
        self.conv1d.fc = self.classifier

    def forward(self, x, len_x):
        x = self.adapter(x)
        x = x.transpose(1, 2)
        conv1d_outputs = self.conv1d(x, len_x)
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])

        return {
            "visual_feat": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs
        }

    def criterion_calculation(self, ret_dict, label, label_lgt, t, T):
        def weight_exp(t, T, alpha=10.0):
            return torch.exp(-alpha * t / T)

        T_weight = weight_exp(t, T)

        loss = self.loss_weights * (T_weight * self.loss['CTCLoss'](
            ret_dict["conv_logits"].log_softmax(-1),
            label.cpu().int(), ret_dict["feat_len"].cpu().int(),
            label_lgt.cpu().int())).mean()

        loss += self.loss_weights * (T_weight * self.loss['CTCLoss'](
            ret_dict["sequence_logits"].log_softmax(-1),
            label.cpu().int(), ret_dict["feat_len"].cpu().int(),
            label_lgt.cpu().int())).mean()

        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
