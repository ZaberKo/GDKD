import torch
import numpy as np
import argparse
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg

from .utils import get_dataset

from tqdm import tqdm

def main(cfg):
    show_cfg(cfg)
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg, use_val_transform=True)

    target_list=[]
    for i, (image, target, index) in tqdm(
        enumerate(train_loader), total=len(train_loader)):
        target_list.append(target)

    target_list=torch.cat(target_list,dim=0).numpy()
    np.save(f"exp/{cfg.DATASET.TYPE}_target2.npy",target_list)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenet")

    args = parser.parse_args()
    if args.dataset == "imagenet":
        cfg_path = "tools/statistics/imagenet.yaml"
    elif args.dataset == "cifar100":
        cfg_path = "tools/statistics/cifar100.yaml"

    cfg.merge_from_file(cfg_path)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg)