# %%

from detectron2.data import MetadataCatalog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import torch
import torch.nn.functional as F

from pathlib import Path

import copy

save_path = Path("./export_img")

# %%

class_names = copy.deepcopy(MetadataCatalog.get("coco_2017_train").thing_classes)
class_names.append("background")

# %%


def read(model):
    logits_dict = np.load(f"calc_logits/num_iter_1000/{model}.npz")

    return logits_dict


def plot_logits_multiclass(model, topk=5, T=1, plot_top1=True, use_log=False, no_bg=False, interest_classes=[0]):
    interest_classes = list(set(interest_classes))
    interest_classes.sort()
    logits_dict = read(model)

    H, W = 9, 10

    if no_bg:
        H-=1

    fig = plt.figure(figsize=(3*W, 3*H), dpi=300)

    for i, ci in enumerate(interest_classes, 1):
        c = f"class{ci}"
        logits = logits_dict[c]
        logits = torch.as_tensor(logits)

        if no_bg:
            logits = logits[:, :-1]

            if i == 81:
                continue

        if use_log:
            probs = F.log_softmax(logits/T, dim=1)
        else:
            probs = F.softmax(logits/T, dim=1)

        probs_avg = probs.mean(axis=0)
        if not plot_top1:
            probs_avg[np.argmax(probs_avg)] = 0
        probs_topk = probs_avg.clone()

        if plot_top1:
            probs_topk[np.argsort(probs_avg)[:-topk]] = 0
            label = f"top{topk}"
        else:
            # top4
            probs_topk[np.argsort(probs_avg)[:-(topk-1)]] = 0
            label = f"top{topk-1}"

        num_cls = probs_avg.shape[0]

        plt.subplot(H, W, i)
        plt.bar(np.arange(num_cls), probs_avg)
        plt.bar(np.arange(num_cls), probs_topk, label=label)
        plt.bar([ci],[probs_avg[ci]], color="red")
        # plt.yscale("log")
        # plt.title(c)

        if i == 1:
            plt.ylabel("Probability", fontsize=24)
            plt.legend(loc="upper right")

        plt.xlabel(class_names[i-1], fontsize=24)

    fig.tight_layout()
    plt.savefig(
        save_path/f'coco_{model}_logits_T{T}{"" if plot_top1 else "_noplot_top1"}.pdf')
    plt.show()


# %%
plot_logits_multiclass(
    model="DKD-R18-R101-iter59999",
    plot_top1=True,
    use_log=False,
    interest_classes=list(range(81))
)
# %%
plot_logits_multiclass(
    model="DKD-R18-R101-iter59999",
    plot_top1=False,
    use_log=False,
    interest_classes=list(range(81))
)
# %%
for i in range(59999, 180000, 60000):
    plot_logits_multiclass(
        model=f"DKD-R18-R101-iter{i}",
        T=4,
        plot_top1=True,
        use_log=False,
        interest_classes=list(range(81))
    )
# %%
for i in range(59999, 180000, 60000):
    plot_logits_multiclass(
        model=f"DKD-R18-R101-iter{i}",
        T=1,
        plot_top1=True,
        use_log=False,
        interest_classes=list(range(81))
    )
# %%
plot_logits_multiclass(
    model="DKD-R18-R101-iter59999",
    plot_top1=True,
    use_log=False,
    no_bg=True,
    interest_classes=list(range(81))
)
# %%
