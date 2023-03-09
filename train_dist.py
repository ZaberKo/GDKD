import argparse

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.utils import log_msg


def run(cmds, gpu_ids):
    cmds = cmds.copy()
    cmds.insert(0, f'CUDA_VISIBLE_DEVICES={",".join(gpu_ids)}')
    cmd_str = " ".join(cmds)
    print(f'Running: {cmd_str}')
    subprocess.run(cmd_str, shell=True, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--ngpu_per_test", type=int, default=1)
    parser.add_argument("--num_tests", type=int, default=1)
    parser.add_argument("--suffix", type=str, nargs="?", default="", const="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    allgpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    allgpu_ids = [int(i) for i in allgpu_ids if i != ""]

    if args.ngpu_per_test > len(allgpu_ids):
        raise ValueError("ngpu_per_test > all gpus")

    print("num_tests:", args.num_tests)
    print("ngpu_per_test:", args.ngpu_per_test)

    cmds = ["python", "-m", "tools.train",
            "--cfg", args.cfg,
            "--group", "--id", "",
            "--record_loss"]
    if args.suffix != "":
        cmds.append("--suffix")
        cmds.append(args.suffix)
    if args.resume:
        cmds.append("--resume")
    cmds.extend(args.opts)

    executor = ProcessPoolExecutor(args.num_tests)

    try:
        gpu_cnt = 0
        tasks = []
        for i in range(args.num_tests):
            _cmds = cmds.copy()
            # id:
            _cmds[7] = str(i)

            gpu_ids = []
            for _ in range(args.ngpu_per_test):
                gpu_ids.append(str(allgpu_ids[gpu_cnt]))
                gpu_cnt = (gpu_cnt+1) % len(allgpu_ids)

            tasks.append(
                executor.submit(run, _cmds, gpu_ids=gpu_ids)
            )

        for future in as_completed(tasks):
            future.result()

    except BaseException as e:
        print(e)
        # mostly handle keyboard interrupt
        print(log_msg("Training failed", "ERROR"))
    finally:
        executor.shutdown(wait=True)
