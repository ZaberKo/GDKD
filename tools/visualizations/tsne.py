#%%
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from sklearn.manifold import TSNE

#%%
def get_tsne(path):
    num_classes = 100
    feats_dict = np.load(path)
    all_features = np.concatenate(list(feats_dict.values()), axis=0)
    all_labels = []
    for i in range(num_classes):
        feat = feats_dict[f"class{i}"]

        all_labels.append(
            np.full((feat.shape[0],), fill_value=i, dtype=np.int32)
        )
    all_labels = np.concatenate(all_labels, axis=0)

    tsne = TSNE()
    reduced_all_features = tsne.fit_transform(all_features)
    plot_features(reduced_all_features, all_labels, num_classes)

def plot_features(features, labels, num_classes):
    # colors = ['C' + str(i) for i in range(num_classes)]
    color_list = np.random.choice(list(colors.CSS4_COLORS), num_classes, replace=False)
    plt.figure(figsize=(6, 6), dpi=300)
    for l in range(num_classes):
        plt.scatter(
            features[labels == l, 0],
            features[labels == l, 1],
            c=color_list[l], s=1, alpha=0.4)
    plt.xticks([])
    plt.yticks([])
    plt.show()
# %%
get_tsne('../../exp/kd_logits_data/dkd_cifar100_MobileNetV2_feats_aug_val.npz')
# %%
get_tsne('../../exp/kd_logits_data/dkd_cifar100_ShuffleV1_feats_aug_val.npz')
# %%
get_tsne('../../exp/kd_logits_data/gdkd_cifar100_ShuffleV1_feats_aug_val.npz')
# %%
