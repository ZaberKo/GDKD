import argparse

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from tools.train import main as train
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.utils import log_msg


def run(cmds, gpu_id):
    cmds = cmds.copy()
    cmds.insert(0, f"CUDA_VISIBLE_DEVICES={gpu_id}")
    cmd_str = " ".join(cmds)
    print(f'Running: {cmd_str}')
    subprocess.run(cmd_str, shell=True, check=True)


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

    gpu_ids=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    gpu_ids=[int(i) for i in gpu_ids]

    gpu_cnt = 0

    print("num_tests:", args.num_tests)

    cmds = ["python", "-m", "tools.train",
            "--cfg", args.cfg, "--group", "--id", ""]
    if args.resume:
        cmds.append("--resume")
    cmds.extend(args.opts)

    executor = ProcessPoolExecutor(args.num_tests)

    try:
        tasks = []
        for i in range(args.num_tests):
            _cmds = cmds.copy()
            _cmds[7] = str(i)

            tasks.append(
                executor.submit(run, _cmds, gpu_id=gpu_ids[gpu_cnt])
            )

            gpu_cnt = (gpu_cnt+1) % len(gpu_ids)

        for future in as_completed(tasks):
            future.result()

    except BaseException as e:
        print(e)
        # mostly handle keyboard interrupt
        print(log_msg("Training failed", "ERROR"))
    finally:
        executor.shutdown(wait=True)
