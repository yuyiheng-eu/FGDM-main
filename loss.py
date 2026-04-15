# coding: utf-8
import torch
import torch.nn as nn

from module.helpers import getSkeletalModelStructure

from einops import rearrange
class Loss(nn.Module):

    def __init__(self, cfg, target_pad=0.0):
        super(Loss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()
        self.bone_loss = cfg["training"]["bone_loss"].lower()
        
        if self.loss == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss == "mse":
            self.criterion = nn.MSELoss()
        else:
            print("Loss not found - revert to default L1 loss")
            self.criterion = nn.L1Loss()

        if self.bone_loss == "l1":
            self.criterion_bone = nn.L1Loss()
        elif self.bone_loss == "mse":
            self.criterion_bone = nn.MSELoss()
        else:
            print("Loss not found - revert to default MSE loss")
            self.criterion_bone = nn.MSELoss()
        
        model_cfg = cfg["model"]

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 0.1)

    def forward(self, preds, targets):

        loss_mask = (targets != self.target_pad)

        # Find the masked predictions and targets using loss mask
        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask

        preds_masked_length, preds_masked_direct = get_length_direct(preds_masked)
        targets_masked_length, targets_masked_direct = get_length_direct(targets_masked)

        preds_masked_length = preds_masked_length * loss_mask[:, :, :50]
        targets_masked_length = targets_masked_length * loss_mask[:, :, :50]
        preds_masked_direct = preds_masked_direct * loss_mask[:, :, :150]
        targets_masked_direct = targets_masked_direct * loss_mask[:, :, :150]


        loss = self.criterion(preds_masked, targets_masked) + \
        0.1 * self.criterion_bone(preds_masked_direct, targets_masked_direct)

        # Multiply loss by the loss scale
        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss

def get_length_direct(trg):
    trg_reshaped = trg.view(trg.shape[0], trg.shape[1], 50, 3)
    trg_list = trg_reshaped.split(1, dim=2)
    trg_list_squeeze = [t.squeeze(dim=2) for t in trg_list]
    skeletons = getSkeletalModelStructure()

    length = []
    direct = []
    for skeleton in skeletons:
        result_length = Skeleton_length = torch.norm(trg_list_squeeze[skeleton[0]]-trg_list_squeeze[skeleton[1]], p=2, dim=2, keepdim=True)
        result_direct = (trg_list_squeeze[skeleton[0]]-trg_list_squeeze[skeleton[1]]) / (Skeleton_length+torch.finfo(Skeleton_length.dtype).tiny)
        direct.append(result_direct)
        length.append(result_length)
    lengths = torch.stack(length, dim=-1).squeeze()
    directs = torch.stack(direct, dim=2).view(trg.shape[0], trg.shape[1], -1)

    return lengths, directs



