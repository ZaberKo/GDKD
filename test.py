#%%
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from mdistiller.models import get_tiny_imagenet_model

#%%
model = get_tiny_imagenet_model("ResNet34", pretrained=False)
model
# %%
from pprint import pprint
state_dict=torch.load("download_ckpts/imagenet_teachers/resnet18/resnet18-5c106cde.pth")
pprint(list(state_dict.keys()))
# %%
from pprint import pprint
state_dict=torch.load("output/imagenet_baselines_ckpt/gdkd,res34,res18_0_2023-03-09-00-27-00/best")
pprint(list(state_dict.keys()))
# %%
pprint(list(state_dict["optimizer"].keys()))
# %%
state_dict
print()
# %%
from pprint import pprint
state_dict=torch.load("download_ckpts/cifar_teachers/ResNet50_aug/ckpt_epoch_240.pth")
pprint(list(state_dict.keys()))
# %%
