# %%
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LogNorm
from pathlib import Path

import sklearn.metrics

save_dir = Path('../../exp/export_img/corr')

if not save_dir.exists():
    save_dir.mkdir(parents=True)


# %%

# visualize the difference between the teacher's output logits and the student's
def get_confusion_matrix(path):
    res = np.load(path)
    preds = res['logits']
    labels = res['labels']

    # preds = torch.softmax(torch.from_numpy(preds)/T, dim=1).numpy()
    matrix = sklearn.metrics.confusion_matrix(labels, preds.argmax(axis=1))

    return matrix


def plot(path):
    # mt = get_confusion_matrix(path_t, T)
    m = get_confusion_matrix(path)

    m = m / m.sum()

    np.fill_diagonal(m, 0)

    # sns.heatmap(m, vmin=0, vmax=1.0, cmap="PuBuGn")
    sns.heatmap(m, cmap="flare")
    plt.xticks([])
    plt.yticks([])
    # sns.heatmap(diff, vmin=diff.min(), vmax=1.0, cmap="PuBuGn", norm=LogNorm())
    plt.show()


# %%
plot(
    '../../exp/kd_logits_data/dkd_cifar100_resnet8x4_aug_val.npz'
)
# %%


def plot_arr(paths):
    fig, axs = plt.subplots(
        1, len(paths)+1, figsize=(3*len(paths)+0.25, 3), dpi=300,
        gridspec_kw=dict(width_ratios=[1]*len(paths)+[0.25/3])
    )

    ms = [get_confusion_matrix(path) for path in paths]
    # ms = [m / m.sum() for m in ms]

    m_max = np.stack(ms).max()

    for i, (m, path) in enumerate(zip(ms, paths), 1):

        _m = m*5
        # _m=m

        np.fill_diagonal(_m, 0)

        ax = axs[i-1]
        if i < len(paths):
            sns.heatmap(_m, vmin=0, vmax=m_max,  cmap="flare",
                        square=True, cbar=False, ax=ax, )
        else:
            sns.heatmap(_m,  vmin=0, vmax=m_max, cmap="flare", square=True,
                        cbar=True, ax=ax, cbar_ax=axs[i], cbar_kws=dict(ticks=[]))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(path.name.split('_')[0])

    # axs[-1].set_ticks([])

    fig.tight_layout()
    plt.show()


plot_arr(
    [
        Path('../../exp/kd_logits_data/kd_cifar100_resnet8x4_aug_val.npz'),
        Path('../../exp/kd_logits_data/dkd_cifar100_resnet8x4_aug_val.npz'),
        Path('../../exp/kd_logits_data/gdkd_cifar100_resnet8x4_aug_val.npz'),
    ]
)

# %%
plot_arr(
    [
        Path('../../exp/kd_logits_data/kd_cifar100_ShuffleV2_aug_val.npz'),
        Path('../../exp/kd_logits_data/dkd_cifar100_ShuffleV2_aug_val.npz'),
        Path('../../exp/kd_logits_data/gdkd_cifar100_ShuffleV2_aug_val.npz'),
    ]
)
# %%
