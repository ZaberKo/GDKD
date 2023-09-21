# %%
import torch

# %%
# data=torch.load("pretrained/r18-r101.pth")
data = torch.load("output_final/DKD-R18-R101_temjbxn9/model_0004999.pth")

print(data)
# %%
import torch
from pathlib import Path

def convert(path):
    state_dict = torch.load(path)
    new_ckpt = {}
    for k, v in state_dict.items():
        
        if "teacher" == k.split('.')[0]:
            print(k)
            new_ckpt[k] = v
        else:
            print(f"{k} -> student.{k}")
            new_ckpt["student."+k] = v

    return new_ckpt

root=Path("pretrained")
for path in Path("pretrained_old").glob("*.pth"):
    new_ckpt = convert(path)
    torch.save(new_ckpt, root/path.name)
# %%
from detectron2.data import MetadataCatalog
meta=MetadataCatalog.get("coco_2017_train")
# %%
meta.thing_classes
# %%
meta.thing_dataset_id_to_contiguous_id
# %%
meta.stuff_classes
# %%
