# coding: utf-8
import torch.nn.functional as F

from module.constants import TARGET_PAD

class Batch:
    def __init__(self, torch_batch, pad_index, model):
        self.src, self.src_lengths = torch_batch.src
        self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.nseqs = self.src.size(0)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None
        self.org_lengths = None

        self.file_paths = torch_batch.file_paths
        self.use_cuda = model.use_cuda
        self.target_pad = TARGET_PAD

        if hasattr(torch_batch, "trg"):
            trg = torch_batch.trg
            trg_lengths = torch_batch.trg.shape[1]
            # trg_input is used for teacher forcing, last one is cut off
            # Remove the last frame for target input, as inputs are only up to frame N-1
            self.trg_input = trg.clone()

            self.trg_lengths = trg_lengths
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg.clone()

            # Target Pad is dynamic, so we exclude the padded areas from the loss computation
            trg_mask = (self.trg_input != self.target_pad).unsqueeze(1)
            self.org_lengths = (self.trg_input.sum(dim=-1) != self.target_pad).sum(dim=1)
            # This increases the shape of the target mask to be even (16,1,120,120) -
            # adding padding that replicates - so just continues the False's or True's
            pad_amount = self.trg_input.shape[1] - self.trg_input.shape[2]
            # Create the target mask the same size as target input
            self.trg_mask = (F.pad(input=trg_mask.double(), pad=(pad_amount, 0, 0, 0), mode='replicate') == 1.0)
            self.ntokens = (self.trg != pad_index).data.sum().item()

        if self.use_cuda:
            self._make_cuda()

    # If using Cuda
    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()

        if self.trg_input is not None:
            self.trg_input = self.trg_input.cuda()
            self.trg = self.trg.cuda()
            self.trg_mask = self.trg_mask.cuda()