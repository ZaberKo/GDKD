# %%
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LogNorm
from pathlib import Path

save_dir = Path('../../exp/export_img/corr')

if not save_dir.exists():
    save_dir.mkdir(parents=True)

# %%

# visualize the difference between the teacher's output logits and the student's


def get_output_metric(path, T, num_classes=100):
    res = np.load(path)
    preds = res['logits']
    labels = res['labels']

    preds = torch.softmax(torch.from_numpy(preds)/T, dim=1).numpy()

    matrix = np.zeros((num_classes, num_classes))
    cnt = np.zeros((num_classes, 1))
    for p, l in zip(preds, labels):
        cnt[l, 0] += 1
        matrix[l] += p
    matrix /= cnt
    return matrix


def get_tea_stu_diff(path_t, path_s, scale=1.0, T=4):
    mt = get_output_metric(path_t, T)
    ms = get_output_metric(path_s, T)

    diff = np.abs((ms - mt))
    print('max(diff):', diff.max())
    print('mean(diff):', diff.mean())

    diff = diff / diff.max() * scale

    np.fill_diagonal(diff, 0)

    sns.heatmap(diff, vmin=0, vmax=1.0, cmap="PuBuGn")
    plt.xticks([])
    plt.yticks([])
    # sns.heatmap(diff, vmin=diff.min(), vmax=1.0, cmap="PuBuGn", norm=LogNorm())
    plt.show()


# %%
get_tea_stu_diff(
    path_t='../../exp/kd_logits_data/teacher_cifar100_resnet32x4_aug_val.npz',
    path_s='../../exp/kd_logits_data/kd_cifar100_resnet8x4_aug_val.npz',
    scale=5,
    T=1.0
)

# %%
get_tea_stu_diff(
    path_t='../../exp/kd_logits_data/teacher_cifar100_resnet32x4_aug_val.npz',
    path_s='../../exp/kd_logits_data/dkd_cifar100_resnet8x4_aug_val.npz',
    scale=5,
    T=1.0
)
# %%
get_tea_stu_diff(
    path_t='../../exp/kd_logits_data/teacher_cifar100_resnet32x4_aug_val.npz',
    path_s='../../exp/kd_logits_data/gdkd_cifar100_resnet8x4_aug_val.npz',
    scale=5,
    T=1.0
)


# %%
get_tea_stu_diff(
    path_t='../../exp/kd_logits_data/teacher_cifar100_resnet32x4_aug_val.npz',
    path_s='../../exp/kd_logits_data/kd_cifar100_ShuffleV2_aug_val.npz',
    max_diff=2.0
)
# %%
get_tea_stu_diff(
    path_t='../../exp/kd_logits_data/teacher_cifar100_resnet32x4_aug_val.npz',
    path_s='../../exp/kd_logits_data/dkd_cifar100_ShuffleV2_aug_val.npz',
    max_diff=2.0
)
# %%
get_tea_stu_diff(
    path_t='../../exp/kd_logits_data/teacher_cifar100_resnet32x4_aug_val.npz',
    path_s='../../exp/kd_logits_data/gdkd_cifar100_ShuffleV2_aug_val.npz',
    max_diff=2.0
)
# %%


def get_tea_stu_diff_arr(path_t, path_s_list, scale=1.0, T=4, fname='corr.pdf'):

    # fig = plt.figure(figsize=(3*len(path_s_list), 3), dpi=300)
    fig, axs = plt.subplots(
        1, len(path_s_list)+1, figsize=(3*len(path_s_list)+0.25, 3), dpi=300,
        gridspec_kw=dict(width_ratios=[1]*len(path_s_list)+[0.25/3])
    )
    for i, path_s in enumerate(path_s_list, 1):
        mt = get_output_metric(path_t, T)
        ms = get_output_metric(path_s, T)

        diff = np.abs((ms - mt))
        print(path_s.name.split('_')[0])
        print(
            f"max: {diff.max():.4f}, min: {diff.min()} mean: {diff.mean():.4f}")
        print("="*20)

        diff = diff / diff.max() * scale

        np.fill_diagonal(diff, 0)

        ax = axs[i-1]
        # plt.subplot(1, len(path_s_list), i)
        if i < len(path_s_list):
            sns.heatmap(diff, vmin=0, vmax=1.0, square=True,
                        cmap="PuBuGn", cbar=False, ax=ax)
        else:
            sns.heatmap(diff, vmin=0, vmax=1.0, square=True,
                        cmap="PuBuGn", cbar=True, ax=ax, cbar_ax=axs[i], cbar_kws=dict(ticks=[]))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(path_s.name.split('_')[0])
        # ax.set_aspect("equal")

        # plt.xticks([])
        # plt.yticks([])
        # plt.title(path_s.name.split('_')[0])

    fig.tight_layout()
    # plt.tight_layout()
    plt.savefig(save_dir/fname)
    plt.show()

# %%


get_tea_stu_diff_arr(
    path_t='../../exp/kd_logits_data/teacher_cifar100_resnet32x4_aug_val.npz',
    path_s_list=[
        Path('../../exp/kd_logits_data/kd_cifar100_resnet8x4_aug_val.npz'),
        Path('../../exp/kd_logits_data/dkd_cifar100_resnet8x4_aug_val.npz'),
        Path('../../exp/kd_logits_data/gdkd_cifar100_resnet8x4_aug_val.npz'),
    ],
    scale=4,
    T=1.0,
    fname='corr_resnet32x4-resnet8x4.pdf'
)
# %%
get_tea_stu_diff_arr(
    path_t='../../exp/kd_logits_data/teacher_cifar100_resnet32x4_aug_val.npz',
    path_s_list=[
        Path('../../exp/kd_logits_data/kd_cifar100_ShuffleV2_aug_val.npz'),
        Path('../../exp/kd_logits_data/dkd_cifar100_ShuffleV2_aug_val.npz'),
        Path('../../exp/kd_logits_data/gdkd_cifar100_ShuffleV2_aug_val.npz'),
    ],
    scale=4,
    T=1.0,
    fname='corr_resnet32x4-shuv2.pdf'
)
# %%
