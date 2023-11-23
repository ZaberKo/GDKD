# %%
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

from pathlib import Path
from tqdm import tqdm

save_dir = Path('../../exp/export_img/tsne')

if not save_dir.exists():
    save_dir.mkdir(parents=True)

# %%


def get_tsne(path, num_classes=100, seed=None):
    res = np.load(path)
    feats = res['feats']
    labels = res['labels']

    tsne = TSNE(random_state=seed)
    reduced_all_features = tsne.fit_transform(feats)

    color_list = np.random.choice(
        list(colors.CSS4_COLORS), num_classes, replace=False).tolist()

    plt.figure(figsize=(3, 3), dpi=300)
    plot_features_sns(reduced_all_features, labels, color_list)
    plt.show()


def plot_features(features, labels, color_list):
    num_classes = len(color_list)
    for l in range(num_classes):
        plt.scatter(
            features[labels == l, 0],
            features[labels == l, 1],
            c=color_list[l], s=1, alpha=0.4)
    plt.xticks([])
    plt.yticks([])


def plot_features_sns(features, labels, color_list, s=1):
    sns.scatterplot(
        x=features[:, 0], y=features[:, 1],
        hue=labels,
        # palette=sns.color_palette("hls", num_classes),
        palette=color_list,
        marker='o',
        s=s,
        legend=False,
        alpha=0.4
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()


# %%
get_tsne('../../exp/kd_logits_data/kd_cifar100_resnet8x4_aug_val.npz')
get_tsne('../../exp/kd_logits_data/dkd_cifar100_resnet8x4_aug_val.npz')
get_tsne('../../exp/kd_logits_data/gdkd_cifar100_resnet8x4_aug_val.npz')
# %%
get_tsne('../../exp/kd_logits_data/kd_cifar100_ShuffleV2_aug_val.npz')
get_tsne('../../exp/kd_logits_data/dkd_cifar100_ShuffleV2_aug_val.npz')
get_tsne('../../exp/kd_logits_data/gdkd_cifar100_ShuffleV2_aug_val.npz')
# %%


def get_tsne_arr(paths, num_classes=100, seed=None, fname='tsne.pdf'):
    color_list = np.random.choice(
        list(colors.CSS4_COLORS), num_classes, replace=False).tolist()

    fig = plt.figure(figsize=(3*len(paths), 3), dpi=300)
    for i, path in tqdm(enumerate(paths, 1), total=len(paths)):
        path = Path(path)
        res = np.load(path)
        feats = res['feats']
        labels = res['labels']

        tsne = TSNE(random_state=seed)
        reduced_all_features = tsne.fit_transform(feats)

        plt.subplot(1, len(paths), i)
        plot_features_sns(reduced_all_features, labels, color_list)
        plt.title(path.name.split('_')[0].upper())
    fig.tight_layout()
    plt.savefig(save_dir/fname)
    plt.show()

#%%

get_tsne_arr([
    '../../exp/kd_logits_data/kd_cifar100_resnet8x4_aug_val.npz',
    '../../exp/kd_logits_data/dkd_cifar100_resnet8x4_aug_val.npz',
    '../../exp/kd_logits_data/gdkd_cifar100_resnet8x4_aug_val.npz',
], fname='tsne_resnet32x4-resnet8x4.pdf')

#%%
get_tsne_arr([
    '../../exp/kd_logits_data/kd_cifar100_ShuffleV2_aug_val.npz',
    '../../exp/kd_logits_data/dkd_cifar100_ShuffleV2_aug_val.npz',
    '../../exp/kd_logits_data/gdkd_cifar100_ShuffleV2_aug_val.npz',
], fname='tsne_resnet32x4-ShuffleV2.pdf')
# %%
