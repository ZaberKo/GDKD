import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os
import torch

from tools.train import main as train
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.utils import log_msg

def run(cfg, resume, opts, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # torch.cuda.set_device()
    try:
        train(cfg, resume, opts, group_flag=True)
    except Exception as e:
        print(log_msg(str(e)))
        if cfg.LOG.WANDB:
            try:
                import wandb
                wandb.finish(exit_code=1)
            except:
                print(log_msg("Failed to use WANDB", "INFO"))

        


if __name__ == "__main__":
    import argparse

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

    print("num_tests: ", args.num_tests)


    pool = mp.Pool(processes=args.num_tests)

    try:
        for i in range(args.num_tests):
            print(f"Start test {i}, use GPU {gpu_ids[gpu_cnt]}")
            pool.apply_async(run, kwds=dict(
                cfg=cfg,
                resume=args.resume,
                opts=args.opts,
                gpu_id=gpu_ids[gpu_cnt]
            ))

            gpu_cnt = (gpu_cnt+1) % len(gpu_ids)
    except Exception:
        pool.terminate()
    finally:
        pool.close()
        pool.join()
        
    



