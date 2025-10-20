import argparse
import torch
import torch.backends.cudnn as cudnn

from mdistiller.engine import validate

cudnn.benchmark = True

from mdistiller.distillers import Vanilla
from mdistiller.models import get_model
from mdistiller.dataset import get_dataset
from mdistiller.dataset.imagenet import get_imagenet_val_loader
from mdistiller.engine.utils import load_checkpoint
from mdistiller.engine.cfg import CFG as cfg


def get_val_dataloader(dataset, batch_size):
    get_imagenet_val_loader(args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "imagenet", "cub2011", "tiny-imagenet"],
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--aug_teacher", action="store_true")
    args = parser.parse_args()


    cfg.DATASET.TYPE = args.dataset
    cfg.DISTILLER.AUG_TEACHER = args.aug_teacher
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size
    cfg.DISTILLER.TYPE = "NONE"

    cfg.freeze()

    if args.ckpt == "pretrain":
        model = get_model(cfg, args.model, pretrained=True)
    else:
        model = get_model(cfg, args.model, pretrained=False)
        model.load_state_dict(load_checkpoint(args.ckpt))

    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    model = Vanilla(model)
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    test_acc, test_acc_top5, test_loss = validate(val_loader, model)
    print(f"test_acc:{test_acc:.4f}, test_acc_top5:{test_acc_top5:.4f}, test_loss:{test_loss:.4f}")
