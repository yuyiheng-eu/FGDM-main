# coding: utf-8
import argparse
import os
from training import train, test
def main():

    ap = argparse.ArgumentParser("Focal–General Diffusion Model with Semantic Consistent Guidance for Sign Language Production")

    # Choose between Train and Test
    ap.add_argument("mode", choices=["train", "test"], help="please select 'train' or 'test'")
    
    # Path to Config
    ap.add_argument("config_path", default="./Configs/FGDM.yaml", type=str, help="path to YAML config file")

    # Optional path to checkpoint
    ap.add_argument("--ckpt", type=str, help="path to model checkpoint")

    ap.add_argument("--gpu_id", type=str, default="0", help="gpu to run your job on")

    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # If Train
    if args.mode == "train":
        train(cfg_file=args.config_path, ckpt=args.ckpt)
    # If Test
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()