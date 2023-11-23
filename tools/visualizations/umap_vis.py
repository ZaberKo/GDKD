# %%
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import umap
import umap.plot
import seaborn as sns
import pandas as pd

from pathlib import Path
from tqdm import tqdm

save_dir = Path('../../exp/export_img/umap')

if not save_dir.exists():
    save_dir.mkdir(parents=True)


# %%


def get_umap(path, num_classes=100, seed=None):
    res = np.load(path)
    feats = res['feats']
    labels = res['labels']

    # tsne = TSNE(random_state=seed)
    umap_model = umap.UMAP(
        n_neighbors=100,
        n_components=2,
        min_dist=0.01,
        random_state=seed
    )
    reduced_all_features = umap_model.fit_transform(feats)
    # plot_features(reduced_all_features, labels, num_classes)

    color_list = np.random.choice(
        list(colors.CSS4_COLORS), num_classes, replace=False).tolist()

    plt.figure(figsize=(6, 6), dpi=300)
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
get_umap('../../exp/kd_logits_data/kd_cifar100_resnet8x4_aug_val.npz')
get_umap('../../exp/kd_logits_data/dkd_cifar100_resnet8x4_aug_val.npz')
get_umap('../../exp/kd_logits_data/gdkd_cifar100_resnet8x4_aug_val.npz')
# %%
get_umap('../../exp/kd_logits_data/kd_cifar100_resnet8x4_aug_val.npz')
# %%

# %%
get_umap('../../exp/kd_logits_data/kd_cifar100_ShuffleV2_aug_val.npz')
get_umap('../../exp/kd_logits_data/dkd_cifar100_ShuffleV2_aug_val.npz')
get_umap('../../exp/kd_logits_data/gdkd_cifar100_ShuffleV2_aug_val.npz')
# %%


def get_umap(path, num_classes=100, seed=None):
    res = np.load(path)
    feats = res['feats']
    labels = res['labels']

    # vals = np.arange(15, 75+1, 5).astype(np.int32)
    # vals = np.linspace(0.1, 1, 10)
    vals = [0.1, 0.01, 0.001]

    w = min(4, len(vals))
    h = int(np.ceil(len(vals) / w))

    color_list = np.random.choice(
        list(colors.CSS4_COLORS), num_classes, replace=False).tolist()

    fig = plt.figure(figsize=(3*w, 3*h), dpi=300)

    for i, v in tqdm(enumerate(vals, 1), total=len(vals)):
        umap_model = umap.UMAP(
            n_neighbors=100,
            n_components=2,
            min_dist=v,
            set_op_mix_ratio=0.1,
            random_state=seed
        )
        reduced_all_features = umap_model.fit_transform(feats)
        plt.subplot(h, w, i)
        plot_features_sns(reduced_all_features, labels, color_list)
        plt.title(str(v))

    fig.tight_layout()
    plt.show()


get_umap('../../exp/kd_logits_data/gdkd_cifar100_resnet8x4_aug_val.npz')
# %%
def get_umap_arr(paths, num_classes=100, seed=None, color_list=None, fname='umap.pdf'):
    if color_list is None:
        color_list = np.random.choice(
            list(colors.CSS4_COLORS), num_classes, replace=False).tolist()


    opts = ['euclidean', 'manhattan', 'cosine', 'correlation']
    # opts = ['euclidean']

    fig = plt.figure(figsize=(3*len(opts), 3*len(paths)), dpi=300)
    # fig, axs = plt.subplots(1, len(paths), figsize=(3*len(paths),3), dpi=300)

    cnt = 1
    
    for path in tqdm(paths, total=len(paths)):
        path = Path(path)
        res = np.load(path)
        feats = res['feats']
        labels = res['labels']

        if num_classes < 100:
            select = labels < num_classes
            feats = feats[select]
            labels = labels[select]

        for opt in opts:

            # tsne = TSNE(random_state=seed)
            umap_model = umap.UMAP(
                n_neighbors=15,
                n_components=2,
                min_dist=0.5,
                # densmap=True,
                # set_op_mix_ratio=0.1,
                metric=opt,
                random_state=seed
            )
            reduced_all_features = umap_model.fit_transform(feats)
            plt.subplot(len(paths), len(opts), cnt)
            plot_features_sns(reduced_all_features, labels, color_list, s=3)
            # ax = axs[i-1]
            # umap.plot.points(umap_model, labels=labels, width=3*300, height=3*300, show_legend=False)

            if cnt <= len(opts):
                plt.title(opt)
            if cnt%len(opts) == 1:
                plt.ylabel(path.name.split('_')[0].upper())

            cnt+=1

    fig.tight_layout()
    plt.savefig(save_dir/fname)
    plt.show()


# get_umap_arr(
#     [
#         '../../exp/kd_logits_data/kd_cifar100_resnet8x4_aug_val.npz',
#         '../../exp/kd_logits_data/dkd_cifar100_resnet8x4_aug_val.npz',
#         '../../exp/kd_logits_data/gdkd_cifar100_resnet8x4_aug_val.npz',
#     ]
# )
#%%
get_umap_arr(
    [
        '../../exp/kd_logits_data/kd_cifar100_ShuffleV2_aug_val.npz',
        '../../exp/kd_logits_data/dkd_cifar100_ShuffleV2_aug_val.npz',
        '../../exp/kd_logits_data/gdkd_cifar100_ShuffleV2_aug_val.npz',
    ],
    num_classes=20,
    color_list=list(plt.cm.tab20.colors),
    fname='umap_ShuffleV2.pdf'
)
# %%
get_umap_arr(
    [
        '../../exp/kd_logits_data/kd_cifar100_resnet8x4_aug_val.npz',
        '../../exp/kd_logits_data/dkd_cifar100_resnet8x4_aug_val.npz',
        '../../exp/kd_logits_data/gdkd_cifar100_resnet8x4_aug_val.npz',
    ],
    fname='umap_resnet32x4-resnet8x4.pdf'
)
# %%
get_umap_arr(
    [
        '../../exp/kd_logits_data/kd_cifar100_ShuffleV2_aug_val.npz',
        '../../exp/kd_logits_data/dkd_cifar100_ShuffleV2_aug_val.npz',
        '../../exp/kd_logits_data/gdkd_cifar100_ShuffleV2_aug_val.npz',
    ],
    fname='umap_resnet32x4-ShuffleV2.pdf'
)
# %%
