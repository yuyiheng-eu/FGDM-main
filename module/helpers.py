# coding: utf-8
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
import yaml
import torch
import numpy as np

from torch import nn, Tensor
from .dtw import dtw
from logging import Logger
from typing import Optional

class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """

def make_model_dir(model_dir: str, overwrite=False, test=False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    # If model already exists
    if os.path.isdir(model_dir):
        if test:
            return model_dir
        ans = input(f'The dir{model_dir} is already exist, do you want to remove and refreash it?\n').lower()
        if ans in ['yes','y','ok','1']:
            for file in os.listdir(model_dir):
                file_path = os.path.join(model_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            shutil.rmtree(model_dir, ignore_errors=True)
        else:
            print('Dir not removed!')
            return model_dir
    
    os.makedirs(model_dir)
    return model_dir







def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    fh = logging.FileHandler(
        "{}/{}".format(model_dir, log_file))
    fh.setLevel(level=logging.DEBUG)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logging.getLogger("").addHandler(sh)
    logger.info("FGDM: Iconicity Disentangled Diffusion for Sign Language Production")
    return logger

def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))

def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')

    return torch.from_numpy(mask) == 0 # Turns it into True and False's

# Subsequent mask of two sizes
def uneven_subsequent_mask(x_size: int, y_size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, x_size, y_size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0  # Turns it into True and False's

def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")

def get_latest_checkpoint(ckpt_dir, post_fix="_every" ) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory, of either every validation step or best
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir: directory of checkpoint
    :param post_fixe: type of checkpoint, either "_every" or "_best"

    :return: latest checkpoint file
    """
    # Find all the every validation checkpoints
    list_of_files = glob.glob("{}/*{}.ckpt".format(ckpt_dir,post_fix))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint

def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    # assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location='cuda' if use_cuda else 'cpu', weights_only=False)
    return checkpoint

def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False

def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def calculate_dtw(references, hypotheses):
    """
    Calculate the DTW costs between a list of references and hypotheses

    :param references: list of reference sequences to compare against
    :param hypotheses: list of hypothesis sequences to fit onto the reference

    :return: dtw_scores: list of DTW costs
    """
    # Euclidean norm is the cost function, difference of coordinates
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    dtw_scores = []

    # Remove the BOS frame from the hypothesis
    # hypotheses = hypotheses[:, 1:]    # Non-autoregressive annotation

    # For each reference in the references list
    for i, ref in enumerate(references):
        # Cut the reference down to the max count value
        _ , ref_max_idx = torch.max(ref[:, -1], 0)
        if ref_max_idx == 0: ref_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        ref_count = ref[:ref_max_idx,:-1].cpu().numpy()

        # Cut the hypothesis down to the max count value
        hyp = hypotheses[i]
        _, hyp_max_idx = torch.max(hyp[:, -1], 0)
        if hyp_max_idx == 0: hyp_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        hyp_count = hyp[:hyp_max_idx,:-1].cpu().numpy()

        # Calculate DTW of the reference and hypothesis, using euclidean norm
        d, cost_matrix, acc_cost_matrix, path = dtw(ref_count, hyp_count, dist=euclidean_norm)

        # Normalise the dtw cost by sequence length
        d = d/acc_cost_matrix.shape[0]

        dtw_scores.append(d)

    # Return dtw scores and the hypothesis with altered timing
    return dtw_scores


def getSkeletalModelStructure():
    return (
        # head
        (1, 0),

        (1, 1),     # 中心

        (1, 2),

        # left arm
        (2, 3),

        (3, 4),

        (1, 5),

        (5, 6),

        (6, 7),     # 舍弃

        (7, 8),

        (8, 9),

        (9, 10),

        (10, 11),

        (11, 12),

        (8, 13),

        (13, 14),

        (14, 15),

        (15, 16),

        (8, 17),

        (17, 18),

        (18, 19),

        (19, 20),

        (8, 21),

        (21, 22),

        (22, 23),

        (23, 24),

        (8, 25),

        (25, 26),

        (26, 27),

        (27, 28),

        (4, 29),

        (29, 30),

        (30, 31),

        (31, 32),

        (32, 33),

        (29, 34),

        (34, 35),

        (35, 36),

        (36, 37),

        (29, 38),

        (38, 39),

        (39, 40),

        (40, 41),

        (29, 42),

        (42, 43),

        (43, 44),

        (44, 45),

        (29, 46),

        (46, 47),

        (47, 48),

        (48, 49),
    )
if __name__ == "__main__":
    print(getSkeletalModelStructure())