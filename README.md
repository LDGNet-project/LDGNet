---
project_name: LDGNet-Project
description: >
  LDGNet: LLMs Debate-Guided Network for Multimodal Sarcasm Detection – Official PyTorch Implementation. 
  Thank you for your interest in our paper.
  If you have any questions, don't hesitate to get in touch with us: 82647084@qq.com.
---

## Installation

Follow these steps to set up the environment:

```bash
conda create -n sarcasm python=3.8
conda activate sarcasm
pip install -r requirements.txt
```

---

## Data Preparation

Download and prepare the datasets from the following sources:

- **HFM Dataset**: A widely used dataset for multimodal sarcasm detection, available on [Mendeley Data](https://data.mendeley.com/datasets/h4ymvy9g8j/1).
- **DMSD-CL Dataset**: Designed to evaluate model generalization, more details can be found in the [paper](https://arxiv.org/html/2312.10493v2).
- **HFM Dataset Processing**: Followed by HKEmodel’s preprocessing approach, available on [GitHub](https://github.com/less-and-less-bugs/HKEmodel).

---

## Training

Use the following scripts to train the LDGNet model. Select the configuration that fits your setup.

### Configuration 1: ResNet and BERT
```bash
cd /path/to/main
export DATA="/path/to/dataset"
export LOG="/path/to/save/logs"

CUDA_VISIBLE_DEVICES=0,1 python run.py with data_root=$DATA \
    num_gpus=2 num_nodes=1 per_gpu_batchsize=4 batch_size=16 \
    image_size=224 vit_randaug max_text_len=512 \
    tokenizer="bert-base" vit="resnet" seed=42 \
    log_dir=$LOG precision=32 max_epoch=15 learning_rate=1e-4
```

### Configuration 2: CLIP and T5
```bash
export DATA="/path/to/twitter/dataset"
export LOG="/path/to/checkpoints"

rm -rf $LOG
mkdir $LOG

CUDA_VISIBLE_DEVICES=0 python run.py with data_root=$DATA \
    num_gpus=1 num_nodes=1 per_gpu_batchsize=6 batch_size=16 \
    clip32_base224 text_t5_base image_size=224 vit_randaug \
    max_text_len=512 seed=42 \
    log_dir=$LOG precision=32 max_epoch=5 learning_rate=1.5e-4
```

---

## Inference

Load the trained model and use the following commands for evaluation and testing.

### Configuration 1: ResNet and BERT
```bash
cd /path/to/main
export DATA="/path/to/test/data"
export LOG="/path/to/logs"

CUDA_VISIBLE_DEVICES=0,1 python run.py with data_root=$DATA \
    num_gpus=2 num_nodes=1 per_gpu_batchsize=4 batch_size=16 test_only=True \
    image_size=224 vit_randaug tokenizer="bert-base" vit="resnet" \
    log_dir=$LOG precision=32 max_text_len=512 \
    load_path="/path/to/checkpoints/model.ckpt"
```

### Configuration 2: CLIP and T5
```bash
export DATA="/path/to/test/data"
export LOG="/path/to/logs"

CUDA_VISIBLE_DEVICES=0 python run.py with data_root=$DATA \
    num_gpus=1 num_nodes=1 per_gpu_batchsize=6 batch_size=16 test_only=True \
    clip32_base224 text_t5_base image_size=224 vit_randaug \
    log_dir=$LOG precision=32 max_text_len=512 \
    load_path="/path/to/checkpoints/model.ckpt"
```


## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{zhou2024ldgnet,
    title={LDGNet: LLMs Debate-Guided Network for Multimodal Sarcasm Detection},
    url={https://github.com/LDGNet-project/LDGNet},
    author={Zhou, Hengyang and Yan, Jinwu and Chen, Yaqing and Hong, Rongman and Zuo, Wenbo and Jin, Keyan},
    publisher={ICASSP},
    year={2025}
}
```

---

## Acknowledgements

This project builds on the strengths of existing multimodal frameworks and advances sarcasm detection using a unique debate-based methodology.

---
