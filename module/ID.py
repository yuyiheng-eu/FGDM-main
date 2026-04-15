# coding: utf-8
import torch

from .helpers import getSkeletalModelStructure

def ID(trg):
    
    trg_reshaped = trg.view(trg.shape[0], trg.shape[1], 50, 3)
    trg_list = trg_reshaped.split(1, dim=2)
    trg_list_squeeze = [t.squeeze(dim=2) for t in trg_list]
    skeletons = getSkeletalModelStructure()
    trg_reshaped_list = []
    for skeleton in skeletons:
        Skeleton_length = torch.norm(trg_list_squeeze[skeleton[0]]-trg_list_squeeze[skeleton[1]], p=2, dim=2, keepdim=True)
        Skeleton_direct = (trg_list_squeeze[skeleton[0]]-trg_list_squeeze[skeleton[1]]) / (Skeleton_length+torch.finfo(Skeleton_length.dtype).tiny)
        trg_reshaped_list.append(torch.cat((trg_list_squeeze[skeleton[1]], Skeleton_length, Skeleton_direct), dim=2))
    trg_super = torch.stack(trg_reshaped_list, dim=-1).reshape(trg.shape[0],trg.shape[1],50*7)

    return trg_super