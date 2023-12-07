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


def get_tsne(path, interest_classes=None, seed=None):
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

    tsne = TSNE(random_state=seed)
    reduced_all_features = tsne.fit_transform(interest_feats)

    # color_list = np.random.choice(
    #     list(colors.CSS4_COLORS), len(interest_classes), replace=False).tolist()
    color_list = sns.color_palette()

    plt.figure(figsize=(6, 6), dpi=300)
    plot_features_sns(reduced_all_features, interest_labels, color_list, s=40)
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


def plot_features_sns(features, labels, color_list, s=1, legend='auto'):
    sns.scatterplot(
        x=features[:, 0], y=features[:, 1],
        hue=[idx_to_class[i] for i in labels],
        # palette=sns.color_palette("hls", num_classes),
        palette=color_list,
        marker='o',
        s=s,
        legend=legend,
        alpha=0.4
    )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()


# %%
get_tsne('../../exp/kd_logits_data2/cifar100_kd_res32x4_res8x4_val.npz',
         interest_classes=[47, 52, 56, 59, 96])
get_tsne('../../exp/kd_logits_data2/cifar100_dkdmod_res32x4_res8x4_val.npz',
         interest_classes=[47, 52, 56, 59, 96])
get_tsne('../../exp/kd_logits_data2/cifar100_gdkd_res32x4_res8x4_val.npz',
         interest_classes=[47, 52, 56, 59, 96])


# %%
get_tsne('../../exp/kd_logits_data2/cifar100_kd_res32x4_shuv1_val.npz',
         interest_classes=[47, 52, 56, 59, 96])
get_tsne('../../exp/kd_logits_data2/cifar100_dkdmod_res32x4_shuv1_val.npz',
         interest_classes=[47, 52, 56, 59, 96])
get_tsne('../../exp/kd_logits_data2/cifar100_gdkd_res32x4_shuv1_val.npz',
         interest_classes=[47, 52, 56, 59, 96])
# %%


def get_tsne_arr(paths, interest_classes=None, seed=None, fname='tsne.pdf'):
    # color_list = np.random.choice(
    #     list(colors.CSS4_COLORS), num_classes, replace=False).tolist()

    color_list = sns.color_palette()

    fig = plt.figure(figsize=(6*len(paths), 6), dpi=300)
    for i, path in tqdm(enumerate(paths, 1), total=len(paths)):
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

        tsne = TSNE(random_state=seed)
        reduced_all_features = tsne.fit_transform(interest_feats)

        plt.subplot(1, len(paths), i)
        if i != len(paths):
            plot_features_sns(reduced_all_features,
                              interest_labels, color_list, s=40, legend=False)
        else:
            plot_features_sns(reduced_all_features,
                              interest_labels, color_list, s=40)

        if len(path.name.split('_')) >= 4:
            kd_method = path.name.split('_')[1]
            if kd_method == 'dkdmod':
                plt.title('DKD')
            else:
                plt.title(path.name.split('_')[1].upper())
        else:
            plt.title('Teacher')
    fig.tight_layout()
    plt.savefig(save_dir/fname)
    plt.show()


# %%
# tree
interest_classes = [47, 52, 56, 59, 96]

#
# fish
interest_classes = [30, 32, 67, 72, 73, 91, 95]

get_tsne_arr([
    f'../../exp/kd_logits_data2/cifar100_{kd}_res32x4_shuv1_val.npz'
    for kd in ['kd', 'dkdmod', 'dist', 'gdkd']
]+['../../exp/kd_logits_data2/cifar100_resnet32x4-aug_val.npz'],
    interest_classes=interest_classes,
    fname='tsne_resnet32x4-ShuffleV2.pdf')
# %%
