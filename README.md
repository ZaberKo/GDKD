# Generalized Decoupled Knowledge Distillation (GDKD)

Official implementation of **[Rethinking Decoupled Knowledge Distillation: A Predictive Distribution Perspective](https://doi.org/10.1109/TNNLS.2025.3639562)** (IEEE TNNLS, 2026).

- Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/11289569) / [DOI](https://doi.org/10.1109/TNNLS.2025.3639562)
- Preprint: [arXiv:2512.04625](https://arxiv.org/abs/2512.04625)

This repository contains the **CNN image classification** code (CIFAR-100, ImageNet, and transfer learning). Other official implementations are maintained separately:

- Semantic segmentation (Cityscapes): [ZaberKo/seg-gdkd](https://github.com/ZaberKo/seg-gdkd)
- ViT image classification: [ZaberKo/vit-gdkd](https://github.com/ZaberKo/vit-gdkd)

This repo is a fork from [megvii-research/mdistiller](https://github.com/megvii-research/mdistiller).

We provide the following new features:

- Advanced `Trainer` support: neater code, detailed distillation record during training, more records in wandb, ...

- New datasets and tasks support: Transfer Learning on numerous datasets (Tiny-ImageNet, CUB-200-2011, ...)

- New algorithms support: GDKD(ours), DKDMod, DIST, LS, and some experimental KD methods.

# Instruction

## CIFAR-100

```shell
# Train the teacher model from scratch 5 times:
python train_dist.py --cfg configs/cifar100/vanilla/vgg13.yaml --num_tests=5 DATASET.ENHANCE_AUGMENT True

# Train GDKD model with some options,
# will auto-split the 5 runs on GPU2, GPU5, GPU7:
CUDA_VISIBLE_DEVICES=2,5,7 python train_dist.py --cfg configs/cifar100/gdkd/wrn40_2_shuv1.yaml --num_tests=5 GDKD.W1 2.0 GDKD.TOPK 5 DISTILLER.AUG_TEACHER True

# Enable experimental KD methods in mdistiller/distillers/experimental:
KD_EXPERIMENTAL=1 python train_dist.py --cfg configs/cifar100/experimental/gdkd_autow_v3/wrn40_2_wrn_16_2.yaml --num_tests=5
```

## ImageNet & Transfer Learning

```shell
# ImageNet
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_LEVEL=PXB torchrun --nproc_per_node 4 --nnodes 1 --master_port 29400 -m tools.train_ddp --cfg configs/imagenet/r34_r18/dist.yaml --group --id 0 --data_workers 16

# Tiny-ImageNet
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 python train_dist.py --cfg configs/TL/tiny-imagenet/r50_mv1/kd.yaml --num_tests=1
```

# Citation

If you find this repo useful, please cite our TNNLS paper:

```bibtex
@article{Zheng_2026,
  title={Rethinking Decoupled Knowledge Distillation: A Predictive Distribution Perspective},
  volume={37},
  ISSN={2162-2388},
  url={https://doi.org/10.1109/TNNLS.2025.3639562},
  DOI={10.1109/TNNLS.2025.3639562},
  number={6},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  publisher={Institute of Electrical and Electronics Engineers (IEEE)},
  author={Zheng, Bowen and Cheng, Ran},
  year={2026},
  month=jun,
  pages={2742--2756}
}
```

# Acknowledgement

- Thanks for DKD. We built this library based on the [DKD's codebase](https://github.com/megvii-research/mdistiller)

- The original DKD's codebase is built on the [CRD's codebase](https://github.com/HobbitLong/RepDistiller) and the [ReviewKD's codebase](https://github.com/dvlab-research/ReviewKD).

- DIST: [DIST's codebase](https://github.com/hunto/DIST_KD)

- Logit Standardization: [LS's codebase](https://github.com/sunshangquan/logit-standardization-KD)

- MLKD: [MLKD's codebase](https://github.com/Jin-Ying/Multi-Level-Logit-Distillation)
