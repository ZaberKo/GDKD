import os
import argparse
from tqdm import tqdm
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.models import get_model

from ..statistics.datasets import get_dataset
from .DKDMod_debug import DKDMod

cudnn.benchmark = True


def main(cfg, args):
    # init dataloader & models
    train_loader, num_classes = get_dataset(
        cfg,
        train=True,
        use_val_transform=False
    )

    model_student = get_model(cfg, cfg.DISTILLER.STUDENT, pretrained=False)
    model_teacher = get_model(cfg, cfg.DISTILLER.TEACHER, pretrained=True)
    distiller = DKDMod(model_student, model_teacher, cfg)

    distiller = distiller.cuda()
    distiller = torch.nn.DataParallel(distiller)

    epochs=list(range(1,241,40))

    suffix=""
    if cfg.DATASET.ENHANCE_AUGMENT:
        suffix += "_aug"
    if cfg.DKDMOD.STRATEGY == "top1":
        suffix += "_top1"

    save_path = Path(args.save_dir)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    for epoch in epochs:
        if epoch != 1:
            distiller_statedict_path = os.path.join(args.model_dir, f"epoch_{epoch}")
            if not os.path.exists(distiller_statedict_path):
                distiller_statedict_path+=".pth"
            print("load state:", distiller_statedict_path)
            state = load_checkpoint(distiller_statedict_path)
            distiller.load_state_dict(state)

        distiller.eval()
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            image, target, index = data
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)

            distiller.module.record_grad(image=image, target=target)

        distiller.module.save_record(
            path=save_path/f"{args.save_name}{suffix}/epoch{epoch}.npz")
        distiller.module.reset_record()


# for training with original python reqs
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str,
                        default="configs/cifar100/dkdmod/res32x4_res8x4.yaml")
    parser.add_argument("--model-dir", type=str, default="cifar100_baselines/dkdmod,res32x4,res8x4|LOG.WANDB:False_bak")
    parser.add_argument("--save-dir", type=str, default="exp/kd_logits_data")
    parser.add_argument("--save-name", type=str, default="resnet32x4_8x4_grad")
    parser.add_argument("--aug", action="store_true", help="Use AutoAug")
    parser.add_argument("--top1", action="store_true", help="Use top1 strategy")
    parser.add_argument("--seed", type=int)
    parser.add_argument("opts", nargs="*")
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.top1:
        print("Use top1 strategy")
        cfg.DKDMOD.STRATEGY = "top1"

    if args.aug:
        print("Use AutoAug")
        cfg.DATASET.ENHANCE_AUGMENT = True
    else:
        cfg.DATASET.ENHANCE_AUGMENT = False

    cfg.freeze()

    if args.seed is not None:
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    main(cfg, args)
