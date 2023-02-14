import argparse

import time
import ray
import os

from tools.train import main as train
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.utils import log_msg

@ray.remote
def run(cfg, resume, opts, worker_id, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # torch.cuda.set_device()
    try:
        train(cfg, resume, opts, group_flag=True)
    except (Exception,KeyboardInterrupt) as e:
        print(log_msg(f"worker {worker_id} fail: {e}", "ERROR"))
    finally:
        if cfg.LOG.WANDB:
            try:
                import wandb
                wandb.finish(exit_code=1)
            except Exception as e:
                print(
                    log_msg(f"worker {worker_id} failed to exit wandb: {e}", "ERROR"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--num_tests", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    gpu_ids = [0]
    gpu_cnt = 0

    print("num_tests:", args.num_tests)

    ray.init(num_cpus=args.num_tests)

    try:
        tasks = []
        for i in range(args.num_tests):
            print(f"Start test {i}, use GPU {gpu_ids[gpu_cnt]}")
            tasks.append(
                run.remote(
                    cfg=cfg,
                    resume=args.resume,
                    opts=args.opts,
                    worker_id=i,
                    gpu_id=gpu_ids[gpu_cnt]
                )
            )
            gpu_cnt = (gpu_cnt+1) % len(gpu_ids)

        # join
        ray.wait(tasks, num_returns=len(tasks))
    except:
        print(log_msg("Training failed", "ERROR"))
    finally:
        for task in tasks:
            ray.cancel(task)
        time.sleep(30)
    ray.shutdown()
