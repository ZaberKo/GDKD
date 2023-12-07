# %%
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import umap
import seaborn as sns
import pandas as pd

from pathlib import Path
from tqdm import tqdm

save_dir = Path('../../exp/export_img/umap')

if not save_dir.exists():
    save_dir.mkdir(parents=True)

class_to_idx = {'apple': 0,
                'aquarium_fish': 1,
                'baby': 2,
                'bear': 3,
                'beaver': 4,
                'bed': 5,
                'bee': 6,
                'beetle': 7,
                'bicycle': 8,
                'bottle': 9,
                'bowl': 10,
                'boy': 11,
                'bridge': 12,
                'bus': 13,
                'butterfly': 14,
                'camel': 15,
                'can': 16,
                'castle': 17,
                'caterpillar': 18,
                'cattle': 19,
                'chair': 20,
                'chimpanzee': 21,
                'clock': 22,
                'cloud': 23,
                'cockroach': 24,
                'couch': 25,
                'crab': 26,
                'crocodile': 27,
                'cup': 28,
                'dinosaur': 29,
                'dolphin': 30,
                'elephant': 31,
                'flatfish': 32,
                'forest': 33,
                'fox': 34,
                'girl': 35,
                'hamster': 36,
                'house': 37,
                'kangaroo': 38,
                'keyboard': 39,
                'lamp': 40,
                'lawn_mower': 41,
                'leopard': 42,
                'lion': 43,
                'lizard': 44,
                'lobster': 45,
                'man': 46,
                'maple_tree': 47,
                'motorcycle': 48,
                'mountain': 49,
                'mouse': 50,
                'mushroom': 51,
                'oak_tree': 52,
                'orange': 53,
                'orchid': 54,
                'otter': 55,
                'palm_tree': 56,
                'pear': 57,
                'pickup_truck': 58,
                'pine_tree': 59,
                'plain': 60,
                'plate': 61,
                'poppy': 62,
                'porcupine': 63,
                'possum': 64,
                'rabbit': 65,
                'raccoon': 66,
                'ray': 67,
                'road': 68,
                'rocket': 69,
                'rose': 70,
                'sea': 71,
                'seal': 72,
                'shark': 73,
                'shrew': 74,
                'skunk': 75,
                'skyscraper': 76,
                'snail': 77,
                'snake': 78,
                'spider': 79,
                'squirrel': 80,
                'streetcar': 81,
                'sunflower': 82,
                'sweet_pepper': 83,
                'table': 84,
                'tank': 85,
                'telephone': 86,
                'television': 87,
                'tiger': 88,
                'tractor': 89,
                'train': 90,
                'trout': 91,
                'tulip': 92,
                'turtle': 93,
                'wardrobe': 94,
                'whale': 95,
                'willow_tree': 96,
                'wolf': 97,
                'woman': 98,
                'worm': 99}
idx_to_class = {v: k for k, v in class_to_idx.items()}


# %%


def get_umap(path, interest_classes=None, seed=None):
    res = np.load(path)
    feats = res['feats']
    labels = res['labels']

    if interest_classes is None:
        interest_feats = feats
        interest_labels = labels
    else:
        idx = np.logical_or.reduce([labels == c for c in interest_classes])
        interest_feats = feats[idx]
        interest_labels = labels[idx]

    # tsne = TSNE(random_state=seed)
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.01,
        random_state=seed
    )
    reduced_all_features = umap_model.fit_transform(interest_feats)
    # plot_features(reduced_all_features, labels, num_classes)

    # color_list = np.random.choice(
    #     list(colors.CSS4_COLORS), num_classes, replace=False).tolist()
    color_list = sns.color_palette()[:len(interest_classes)]

    plt.figure(figsize=(6, 6), dpi=300)
    plot_features_sns(reduced_all_features, interest_labels, color_list, s=40)
    plt.show()

def plot_features_sns(features, labels, color_list, s=1, legend='auto'):
    sns.scatterplot(
        x=features[:, 0], y=features[:, 1],
        hue=[idx_to_class[i] for i in labels],
        # palette=sns.color_palette("hls", num_classes),
        palette=color_list,
        marker='o',
        s=s,
        legend=legend,
        alpha=0.5,
        lw=0
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()



# %%
get_umap('../../exp/kd_logits_data2/cifar100_kd_res32x4_res8x4_val.npz',
         interest_classes=[47, 52, 56, 59, 96])
get_umap('../../exp/kd_logits_data2/cifar100_dkdmod_res32x4_res8x4_val.npz',
         interest_classes=[47, 52, 56, 59, 96])
get_umap('../../exp/kd_logits_data2/cifar100_dist_res32x4_res8x4_val.npz',
         interest_classes=[47, 52, 56, 59, 96])
get_umap('../../exp/kd_logits_data2/cifar100_gdkd_res32x4_res8x4_val.npz',
         interest_classes=[47, 52, 56, 59, 96])
# %%
# ======== test different hyperparameters ========
def get_umap_arr(path, interest_classes=None, seed=None):
    res = np.load(path)
    feats = res['feats']
    labels = res['labels']

    if interest_classes is None:
        interest_feats = feats
        interest_labels = labels
    else:
        idx = np.logical_or.reduce([labels == c for c in interest_classes])
        interest_feats = feats[idx]
        interest_labels = labels[idx]

    vals = np.arange(15, 75+1, 5).astype(np.int32)
    # vals = np.linspace(0.1, 1, 10)
    # vals = [0.1, 0.01, 0.001]

    w = min(4, len(vals))
    h = int(np.ceil(len(vals) / w))

    # color_list = np.random.choice(
    #     list(colors.CSS4_COLORS), num_classes, replace=False).tolist()
    color_list = sns.color_palette()[:len(interest_classes)]

    fig = plt.figure(figsize=(6*w, 6*h), dpi=300)

    for i, v in tqdm(enumerate(vals, 1), total=len(vals)):
        umap_model = umap.UMAP(
            n_neighbors=v,
            n_components=2,
            min_dist=0.1,
            set_op_mix_ratio=0.1,
            random_state=seed
        )
        reduced_all_features = umap_model.fit_transform(interest_feats)
        plt.subplot(h, w, i)
        if i == 1:
            plot_features_sns(reduced_all_features, interest_labels, color_list, s=40)
        else:
            plot_features_sns(reduced_all_features, interest_labels, color_list, s=40, legend=False)
        plt.title(str(v))

    fig.tight_layout()
    plt.show()


get_umap_arr('../../exp/kd_logits_data2/cifar100_gdkd_res32x4_res8x4_val.npz', interest_classes=[47, 52, 56, 59, 96])
# %%
def get_umap_arr(paths, interest_classes=None, seed=None, color_list=None, fname='umap.pdf'):
    if color_list is None:
        # color_list = np.random.choice(
        #     list(colors.CSS4_COLORS), num_classes, replace=False).tolist()
        color_list = sns.color_palette()[:len(interest_classes)]


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

        if interest_classes is None:
            interest_feats = feats
            interest_labels = labels
        else:
            idx = np.logical_or.reduce([labels == c for c in interest_classes])
            interest_feats = feats[idx]
            interest_labels = labels[idx]

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
            reduced_all_features = umap_model.fit_transform(interest_feats)
            plt.subplot(len(paths), len(opts), cnt)
            plot_features_sns(reduced_all_features, interest_labels, color_list, s=3)
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
