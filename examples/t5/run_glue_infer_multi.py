import argparse
import os
import shutil
import subprocess
import zipfile

TASK_2_PATH = {
    "mnli_m": "MNLI-bin",
    "mnli_mm": "MNLI-bin",
    "qqp": "QQP-bin",
    "qnli": "QNLI-bin",
    "sst_2": "SST-2-bin",
    "cola": "CoLA-bin",
    "sts_b": "STS-B-bin",
    "mrpc": "MRPC-bin",
    "rte": "RTE-bin"
}

TASK_2_SUBMIT_NAME = {
    "mnli_m": "MNLI-m.tsv",
    "mnli_mm": "MNLI-mm.tsv",
    "qqp": "QQP.tsv",
    "qnli": "QNLI.tsv",
    "sst_2": "SST-2.tsv",
    "cola": "CoLA.tsv",
    "sts_b": "STS-B.tsv",
    "mrpc": "MRPC.tsv",
    "rte": "RTE.tsv"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("ckpt_root", type=str)
    parser.add_argument("cfg_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("cola_cfg", type=str)
    parser.add_argument("cola_ckpt_suffix", type=str)
    parser.add_argument("mnli_m_cfg", type=str)
    parser.add_argument("mnli_m_ckpt_suffix", type=str)
    parser.add_argument("mnli_mm_cfg", type=str)
    parser.add_argument("mnli_mm_ckpt_suffix", type=str)
    parser.add_argument("mrpc_cfg", type=str)
    parser.add_argument("mrpc_ckpt_suffix", type=str)
    parser.add_argument("qnli_cfg", type=str)
    parser.add_argument("qnli_ckpt_suffix", type=str)
    parser.add_argument("qqp_cfg", type=str)
    parser.add_argument("qqp_ckpt_suffix", type=str)
    parser.add_argument("rte_cfg", type=str)
    parser.add_argument("rte_ckpt_suffix", type=str)
    parser.add_argument("sst_2_cfg", type=str)
    parser.add_argument("sst_2_ckpt_suffix", type=str)
    parser.add_argument("sts_b_cfg", type=str)
    parser.add_argument("sts_b_ckpt_suffix", type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with zipfile.ZipFile(os.path.join(args.output_dir, "submit.zip"), "w") as zf:
        for task in ["cola", "mnli_m", "mnli_mm", "mrpc", "qnli", "qqp", "rte", "sst_2", "sts_b"]:
            data_dir = os.path.join(args.data_dir, task)
            if not os.path.exists(data_dir):
                data_dir = os.path.join(args.data_dir, TASK_2_PATH[task])
            ckpt_path = os.path.join(
                args.ckpt_root, getattr(args, task + "_cfg"), "checkpoints",
                f"checkpoint{getattr(args, task + '_ckpt_suffix')}.pt"
            )
            run_dir = os.path.join(args.output_dir, getattr(args, task + "_cfg"))
            gen_subset = "test"
            if task == "mnli_mm":
                gen_subset = "test1"
            p = subprocess.run([
                "python", "examples/t5/t5_sentence_prediction_inference.py",
                "--config-dir", args.cfg_dir,
                "--config-name", getattr(args, task + "_cfg"),
                f"task.data={data_dir}",
                f"common_eval.path={ckpt_path}",
                f"hydra.run.dir={run_dir}",
                "dataset.num_workers=0",
                f"dataset.gen_subset={gen_subset}"
            ])
            if p.returncode != 0:
                shutil.rmtree(run_dir, ignore_errors=True)
            zf.write(os.path.join(run_dir, "predictions.tsv"), TASK_2_SUBMIT_NAME[task], zipfile.ZIP_DEFLATED)


if __name__ == '__main__':
    main()
