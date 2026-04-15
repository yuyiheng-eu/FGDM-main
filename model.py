# coding: utf-8
import torch.nn as nn
import torch
from torch import Tensor
from module.encoder import Encoder
from module.Diffusion import FGDM
from batch import Batch
from module.embeddings import Embeddings
from vocabulary import Vocabulary
from initialization import initialize_model
from module.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, TARGET_PAD
from module.scg_network import SCGModel 
class Model(nn.Module):
    def __init__(self, cfg: dict,
                 encoder: Encoder,
                 diffusion: FGDM,
                 scg_decoder: SCGModel,
                 src_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary,
                 in_trg_size: int,
                 out_trg_size: int):
        """
        Create FGDM

        :param encoder: encoder
        :param diffusion: diffusion model
        :param scg_decoder: SCG decoder
        :param src_embed: source embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        :param in_trg_size: input target size
        :param out_trg_size: output target size
        """
        super(Model, self).__init__()

        self.src_embed = src_embed
        self.encoder = encoder
        self.diffusion = diffusion
        self.scg_decoder = scg_decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        # Constants
        self.bos_index = src_vocab.stoi[BOS_TOKEN]
        self.pad_index = src_vocab.stoi[PAD_TOKEN]
        self.eos_index = src_vocab.stoi[EOS_TOKEN]
        self.target_pad = TARGET_PAD

        # Training config
        self.use_cuda = cfg["training"]["use_cuda"]
        self.use_scg = cfg["training"]["use_scg"]

        # Size parameters
        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size

    def forward(self, is_train: bool, src: Tensor, trg_input: Tensor,
                src_mask: Tensor, src_lengths: Tensor, trg_mask: Tensor,
                len_x=None):
        """
        Encode source sequence and decode target.

        :param is_train: training mode flag
        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :param len_x: optional sequence lengths for SCG decoder
        :return: diffusion output (and SCG outputs in training mode)
        """
        # Encode the source sequence
        encoder_output = self.encoder(
            embed_src=self.src_embed(src),
            src_length=src_lengths,
            mask=src_mask
        )

        # Diffusion forward/denoise
        diffusion_result = self.diffusion(
            encoder_output=encoder_output,
            input_3d=trg_input,
            src_mask=src_mask,
            trg_mask=trg_mask,
            is_train=is_train
        )

        if not is_train:
            return diffusion_result

        # Training mode: unpack results and optionally run SCG decoder
        diffusion_output, x_scg, t, T = diffusion_result
        ret_dict = self.scg_decoder(x_scg, len_x) if self.use_scg else {}

        return diffusion_output, ret_dict, t, T, encoder_output

 
    
    def get_loss_for_batch(self, is_train, batch: Batch, loss_function: nn.Module) -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # Forward through the batch input
        skel_out, ret_dict, t, T, _ = self.forward(src=batch.src,
                                trg_input=batch.trg_input[:, :, :150],
                                src_mask=batch.src_mask,
                                src_lengths=batch.src_lengths,
                                trg_mask=batch.trg_mask,
                                is_train=is_train,
                                len_x=batch.org_lengths)

        if self.use_scg:
            label, label_lgt = clean_labels_and_lengths(batch.src, batch.src_lengths)
            scg_loss = self.scg_decoder.criterion_calculation(ret_dict, label, label_lgt, t, T)

            # compute batch loss using skel_out and the batch target
            batch_loss = loss_function(skel_out, batch.trg_input[:, :, :150]) + scg_loss
            
        else:
            
            batch_loss = loss_function(skel_out, batch.trg_input[:, :, :150])

        # return batch loss = sum over all elements in batch that are not pad
        return batch_loss
def clean_labels_and_lengths(labels, lengths, remove_tokens={2, 3}):
    """
    labels: [B, L] LongTensor
    lengths: [B] LongTensor
    remove_tokens: set of tokens to remove (pad=1, eos=3, etc.)
    """
    B, L = labels.shape
    new_labels = []
    new_lengths = []

    for i in range(B):
        # 取出该样本的有效部分（按原长度截断）
        seq = labels[i, :lengths[i]]

        filtered = []
        for tok in seq.tolist():
            if tok in remove_tokens:
                continue
            # 做 remap
            if tok == 0:
                tok = 1  # 0 → 1
            else:
                tok = tok - 3 + 1  # 其它类别整体偏移
            filtered.append(tok)
        new_labels.extend(filtered)
        new_lengths.append(len(filtered))
    return torch.tensor(new_labels, dtype=torch.long), torch.tensor(new_lengths, dtype=torch.long)
def build_model(cfg: dict, src_vocab: Vocabulary, trg_vocab: Vocabulary):

    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    full_cfg = cfg
    cfg = cfg["model"]

    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = 0

    # Input target size is the joint vector length plus one for counter
    in_trg_size = cfg["trg_size"]
    # Output target size is the joint vector length plus one for counter
    out_trg_size = cfg["trg_size"]

    # Define source embedding,vocab_size=1089
    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)
    
    ## Encoder -------
    enc_dropout = cfg["encoder"].get("dropout", 0.) # Dropout
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
           cfg["encoder"]["hidden_size"], \
           "for transformer, emb_size must be hidden_size"
    
    # Transformer Encoder
    encoder = Encoder(**cfg["encoder"],
                      emb_size=src_embed.embedding_dim,
                      emb_dropout=enc_emb_dropout)
    
    # FGDM Diffusion Model
    diffusion = FGDM(args=cfg,
                    trg_vocab=trg_vocab)

    # SCG Decoder
    scg_decoder = SCGModel(cfg=cfg["scgmodel"], num_classes=len(src_vocab)-3+1)

    # Define the model
    model = Model(encoder=encoder,
                  diffusion=diffusion,
                  scg_decoder=scg_decoder,
                  src_embed=src_embed,
                  src_vocab=src_vocab,
                  trg_vocab=trg_vocab,
                  cfg=full_cfg,
                  in_trg_size=in_trg_size,
                  out_trg_size=out_trg_size)

    # Custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model