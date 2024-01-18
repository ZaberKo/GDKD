# Generalized Decoupled Knowledge Distillation (GDKD)

This repo is a fork from [megvii-research/mdistiller](https://github.com/megvii-research/mdistiller).

We provide the following new features:

- Advanced `Trainer` support: neater code, detailed distillation record during training, more records in wandb, ...

- New datasets and tasks support: Transfer Learning on numerious dataset (Tiny-ImageNet, CUB-200-2011, ...)

- New algorithms support: GDKD(ours), DKDMod, DIST and some experimental KD methods.

# Instruction

## CIFAR-100

```shell
# Train the teacher model from scratch 5 times:
python train_dist.py --cfg configs/cifar100/vanilla/vgg13.yaml --num_tests=5 DATASET.ENHANCE_AUGMENT True

# Train GDKD model with some options,
# will auto-split the 5 runs on GPU2, GPU5, GPU7:
CUDA_VISIBLE_DEVICES=2,5,7 python train_dist.py --cfg configs/cifar100/gdkd/wrn40_2_shuv1.yaml --num_tests=5 GDKD.W1 2.0 GDKD.TOPK 5 DISTILLER.AUG_TEACHER True

# Train experimental model:
KD_EXPERIMENTAL=1 python train_dist.py --cfg configs/cifar100/experimental/gdkd_autow_v3/wrn40_2_wrn_16_2.yaml --num_tests=5
```

## ImageNet & Transfer Learning

```shell
# ImageNet
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_LEVEL=PXB torchrun --nproc_per_node 4 --nnodes 1 --master_port 29400 -m tools.train_ddp --cfg configs/imagenet/r34_r18/dist.yaml --group --id 1 --data_workers 16

# Tiny-ImageNet
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 python train_dist.py --cfg configs/TL/tiny-imagenet/r50_mv1/kd.yaml --num_tests=1
```

# Acknowledgement

- Thanks for DKD. We build this library based on the [DKD's codebase](https://github.com/megvii-research/mdistiller)

- Thanks for CRD and ReviewKD. The original DKD's codebase is built on the [CRD's codebase](https://github.com/HobbitLong/RepDistiller) and the [ReviewKD's codebase](https://github.com/dvlab-research/ReviewKD).

- Thanks for DIST. [DIST's codebase](https://github.com/hunto/DIST_KD)

