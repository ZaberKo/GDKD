import torch
import torch.nn as nn
from tqdm import tqdm

from mdistiller.engine.utils import AverageMeter, Timer, accuracy, log_msg


def validate(val_loader, distiller):
    eval_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(range(len(val_loader)))

    distiller.eval()
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            with Timer() as eval_timer:
                image, target = data[:2]
                image = image.float()
                image = image.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                output = distiller(image=image)
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                batch_size = image.size(0)

                losses.update(loss.item(), batch_size).all_reduce()
                top1.update(acc1.item(), batch_size).all_reduce()
                top5.update(acc5.item(), batch_size).all_reduce()

            # measure elapsed time
            eval_time.update(eval_timer.interval)

            msg = f"Time(data):{eval_time.avg:.3f}| Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}"
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    return top1.avg, top5.avg, losses.avg