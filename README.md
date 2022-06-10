# Efficient Large LM Trainer

This repository contains pretraining pipeline of sequence-to-sequence language models.

We provide scripts based on the [Fairseq](https://github.com/facebookresearch/fairseq) library and PyTorch.

## T5 Pretraining

### Requirements

This codebase requires CUDA 11.3+, Python 3.8+, and PyTorch 1.10.2+. For best compatibility, you can run the following script in a clean Python 3.8 virtual environment:

```bash
python -m pip install torch==1.10.2+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Another dependency is [Fairseq](https://github.com/facebookresearch/fairseq). 

```bash
git clone git@github.com:facebookresearch/fairseq
cd fairseq
git checkout 11b2830d29aed8043e5011d64e14004347a08b50
python -m pip install -e .
```

### Data

Please refer to

### T5 Pretraining

The following pretraining script for pretraining T5-Base on the Wikibook dataset is tested on a node of 8 NVIDIA A100 40GB GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fairseq-hydra-train -m --config-dir examples/t5/config/pretraining \
--config-name t5_base_8gpus \
common.user_dir=$(pwd)/efficent_large_lm_trainer \
task.data=/path/to/wikibook_data \
hydra.sweep.dir=/path/to/outputs
```

Pretraining on a single node will take ~136 hours. We recommend pretraining on 8 nodes. Assuming the `NODE_RANK` environment variable is set to _i_ on the _i_-th node, here is the pretraining script on 8 nodes with 8 GPUs each:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fairseq-hydra-train -m --config-dir examples/t5/config/pretraining \
--config-name t5_base_64gpus \
common.user_dir=$(pwd)/efficent_large_lm_trainer \
task.data=/path/to/wikibook_data \
distributed_training.distributed_world_size=64 \
distributed_training.distributed_rank=$((NODE_RANK * 8)) \
distributed_training.distributed_init_method="tcp://${MASTER_IP}:${MASTER_PORT}" \
hydra.sweep.dir=/path/to/outputs
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
