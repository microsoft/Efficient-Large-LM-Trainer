import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
from glob import glob

TASKS = ["mnli", "qqp", "qnli", "sst_2", "cola", "sts_b", "mrpc", "rte"]

TASK_2_PATH = {
    "mnli": "MNLI-bin",
    "qqp": "QQP-bin",
    "qnli": "QNLI-bin",
    "sst_2": "SST-2-bin",
    "cola": "CoLA-bin",
    "sts_b": "STS-B-bin",
    "mrpc": "MRPC-bin",
    "rte": "RTE-bin"
}

RESULT_RE = re.compile(r"\[[^\[\]]+\]\[valid\]\[INFO\] - ({.*})")
RESULT_1_RE = re.compile(r"\[[^\[\]]+\]\[valid1\]\[INFO\] - ({.*})")
DONE_RE = re.compile(r"\[[^\[\]]+\]\[[^\[\]]+\.train\]\[INFO\] - done training .*")


def read_log(log_path):
    r = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            m = re.fullmatch(RESULT_RE, line)
            if m is not None:
                try:
                    rr = json.loads(m.group(1))
                except json.JSONDecodeError:
                    print("JSON decode error:", log_path)
                    return None
                r.append(rr)
            m = re.fullmatch(RESULT_1_RE, line)
            if m is not None:
                try:
                    rr = json.loads(m.group(1))
                except json.JSONDecodeError:
                    print("JSON decode error:", log_path)
                    return None
                r.append(rr)
            m = re.fullmatch(DONE_RE, line)
            if m is not None:
                return r
    print("Incomplete log:", log_path)
    return None


def maybe_read_log(run_dir):
    if os.path.exists(run_dir):
        log_path = os.path.join(run_dir, "hydra_train.log")
        if os.path.exists(log_path):
            return read_log(log_path)
    return None


q_gpu = mp.Queue()
args = None
cur_gpu = None


def run(job):
    global q_gpu, args, cur_gpu
    task, cfg_name = job
    if cur_gpu is None:
        cur_gpu = q_gpu.get()
    new_env = os.environ.copy()
    new_env["CUDA_VISIBLE_DEVICES"] = str(cur_gpu)
    run_dir = os.path.join(args.output_dir, cfg_name)
    log = maybe_read_log(run_dir)
    while log is None:
        os.makedirs(run_dir, exist_ok=True)
        p = subprocess.run([
            "fairseq-hydra-train",
            "--config-dir", args.cfg_dir,
            "--config-name", cfg_name,
            f"task.data={os.path.abspath(os.path.join(args.data_dir, TASK_2_PATH[task]))}",
            f"checkpoint.restore_file={os.path.abspath(args.ckpt_path)}",
            f"hydra.run.dir={run_dir}"
        ],
            stdout=subprocess.DEVNULL,
            env=new_env
        )
        if p.returncode != 0:
            shutil.rmtree(run_dir, ignore_errors=True)
        log = maybe_read_log(run_dir)
    return {
        "task": task,
        "cfg_name": cfg_name,
        "log": log
    }


def collect_log_only(job):
    task, cfg_name = job
    run_dir = os.path.join(args.output_dir, cfg_name)
    log = maybe_read_log(run_dir)
    assert log is not None
    return {
        "task": task,
        "cfg_name": cfg_name,
        "log": log
    }


def main():
    global q_gpu, args

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("cfg_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--n_gpus", type=int, default=1)
    args = parser.parse_args()

    jobs = []
    for task in TASKS:
        for cfg_path in sorted(glob(os.path.join(args.cfg_dir, f"{task}_*.yaml"))):
            cfg_name = os.path.basename(cfg_path)[:-len(".yaml")]
            jobs.append((task, cfg_name))
    jobs = jobs[args.rank::args.world_size]

    unfinished_jobs = []
    for task, cfg_name in jobs:
        run_dir = os.path.join(args.output_dir, cfg_name)
        log = maybe_read_log(run_dir)
        if log is None:
            shutil.rmtree(run_dir, ignore_errors=True)
            unfinished_jobs.append((task, cfg_name))
    print(f"Detected {len(unfinished_jobs)} unfinished jobs")

    for i in range(args.n_gpus):
        q_gpu.put(i)
    with mp.Pool(args.n_gpus) as pool:
        pool.map(run, unfinished_jobs)
    with mp.Pool(args.n_gpus) as pool:
        all_logs = pool.map(collect_log_only, jobs)
    with open(os.path.join(args.output_dir, f"logs_{args.rank}.json"), "w", encoding="utf-8") as f:
        json.dump(all_logs, f)


if __name__ == '__main__':
    main()
