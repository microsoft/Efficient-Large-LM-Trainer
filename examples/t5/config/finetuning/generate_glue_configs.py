import argparse
import json
import os

import yaml

N_EXAMPLES = {
    "mnli": 392702,
    "qqp": 390965,
    "qnli": 104743,
    "sst_2": 67349,
    "cola": 8551,
    "sts_b": 5749,
    "mrpc": 3668,
    "rte": 2490
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tokenizer_args", type=str, default=None)
    parser.add_argument("--bpe", type=str, default=None)
    parser.add_argument("--bpe_args", type=str, default=None)
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--model_override", type=str, default="{}")
    parser.add_argument("--warmup", type=float, default=0.06)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for task in N_EXAMPLES.keys():
        with open(os.path.join(args.input_dir, f"{task}.yaml"), "r", encoding="utf-8") as f:
            m = yaml.safe_load(f)
        if args.tokenizer:
            m['tokenizer'] = {'_name': args.tokenizer}
            if args.tokenizer_args:
                m['tokenizer'].update(**json.loads(args.tokenizer_args))
        if args.bpe:
            m['bpe'] = {'_name': args.bpe}
            if args.bpe_args:
                m['bpe'].update(**json.loads(args.bpe_args))
        if args.arch:
            m['model']['_name'] = args.arch
        for mk, mv in json.loads(args.model_override).items():
            m['model'][mk] = mv
        for lr in [1e-05, 2e-05, 3e-05, 4e-05, 5e-05]:
            for n_epoch in [2, 3, 5, 10]:
                for bs in [16, 32]:
                    for seed in [1, 2, 3, 4, 5]:
                        steps_per_epoch = (N_EXAMPLES[task] + bs - 1) // bs
                        max_steps = steps_per_epoch * n_epoch
                        warmup_steps = int(max_steps * args.warmup + 0.5)
                        m['dataset']['batch_size'] = bs
                        m['optimization']['lr'] = [lr]
                        m['optimization']['max_update'] = max_steps
                        m['optimization']['max_epoch'] = n_epoch
                        m['lr_scheduler']['warmup_updates'] = warmup_steps
                        m['common']['seed'] = seed
                        m['checkpoint']['no_save'] = args.no_save
                        with open(
                            os.path.join(
                                args.output_dir,
                                f"{task}_lr{lr}_epoch{n_epoch}_bs{bs}_seed{seed}.yaml"
                            ), "w", encoding="utf-8"
                        ) as f:
                            yaml.dump(m, f)


if __name__ == '__main__':
    main()
