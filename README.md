# ExplainHM
Official PyTorch implementation for the paper - **Towards Explainable Harmful Meme Detection through Multimodal Debate between Large Language Models**.

(**WWW 2024**: The ACM Web Conference 2024, May 2024, Singapore.) [[`paper`](https://arxiv.org/pdf/2401.13298.pdf)]


## Install

```bash
conda create -n meme python=3.8
conda activate meme
pip install -r requirements.txt
```

## Data

Please refer to [data](https://github.com/HKBUNLP/ExplainHM-WWW2024/tree/main/data).

## Training
```使用bert和resnet进行训练```
cd /root/main
export DATA="/root/autodl-tmp/HFM"
export LOG="/root/autodl-tmp/ours"

CUDA_VISIBLE_DEVICES=0,1 python run.py with data_root=$DATA \
    num_gpus=2 num_nodes=1 task_train per_gpu_batchsize=4 batch_size=16 \
    image_size=224 vit_randaug max_text_len=512 \
    tokenizer="google" vit="clip" seed=42 \
    log_dir=$LOG precision=32 max_epoch=15 learning_rate=1e-4
``` ```
OR
```使用google-flan和clip进行训练```
export DATA="path/twitter"
export LOG="path/to/save/ckpts/name"

rm -rf $LOG
mkdir $LOG

CUDA_VISIBLE_DEVICES=0 python run.py with data_root=$DATA \
    num_gpus=1 num_nodes=1 task_train per_gpu_batchsize=6 batch_size=6 \
    clip32_base224 text_t5_base image_size=224 vit_randaug max_text_len=512 \
    seed=42 \
    log_dir=$LOG precision=32 max_epoch=5 learning_rate=1.5e-4
``` ```
## Inference
```使用bert和resnet进行测试```
cd /root/main
export DATA="/root/autodl-tmp/HFM"
export LOG="path/to/log/folder"

CUDA_VISIBLE_DEVICES=0,1 python run.py with data_root=$DATA \
    num_gpus=2 num_nodes=1 task_train per_gpu_batchsize=4 batch_size=16 test_only=True \
    image_size=224 vit_randaug \
    tokenizer="google" vit="clip" \
    log_dir=$LOG precision=32 \
    max_text_len=512 load_path="/root/autodl-tmp/ours/MEME_seed42_from_/version_22/checkpoints/epoch=1-step=4909.ckpt"
``` ```
OR
```使用google-flan和clip进行测试 ```
export DATA="path/to/data/folder"
export LOG="path/to/log/folder"

CUDA_VISIBLE_DEVICES=0 python run.py with data_root=$DATA \
    num_gpus=1 num_nodes=1 task_train per_gpu_batchsize=6 batch_size=6 test_only=True \
    clip32_base224 text_t5_base image_size=224 vit_randaug \
    log_dir=$LOG precision=32 \
    max_text_len=512 load_path="path/to/save/ckpts/name/MEME_seed41_from_/version_0/checkpoints/epoch=1-step=1005.ckpt"
``` ```
## Citation

```
@inproceedings{lin2024explainable,
    title={Towards Explainable Harmful Meme Detection through Multimodal Debate between Large Language Models},
    author={Hongzhan Lin and Ziyang Luo and Wei Gao and Jing Ma and Bo Wang and Ruichao Yang},
    booktitle={The ACM Web Conference 2024},
    year={2024},
    address={Singapore},
}
```

## Acknowledgements

The code is based on [ViLT](https://github.com/dandelin/ViLT) and [METER](https://github.com/zdou0830/METER/tree/main).

