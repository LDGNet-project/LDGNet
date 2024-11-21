---
# LDGNet-project

Official PyTorch implementation of **LDGNet: LLMs Debate-Guided Network for Multimodal Sarcasm Detection**. This novel framework introduces a multimodal debate mechanism between large language models (LLMs) to leverage open-world knowledge and enhance sarcasm detection in multimodal datasets.

## Install

To set up the environment, follow these steps:

```bash
conda create -n sarcasm python=3.8
conda activate sarcasm
pip install -r requirements.txt
```

## Data

You can access and prepare the datasets from the [Data Repository](link-to-dataset).

## Training

To train the LDGNet model, use the following scripts tailored for different configurations:

**Configuration 1: Using ResNet and BERT**  
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

**Configuration 2: Using CLIP and T5**  
```bash
export DATA="/path/to/twitter/dataset"
export LOG="/path/to/checkpoints"

rm -rf $LOG
mkdir $LOG

CUDA_VISIBLE_DEVICES=0 python run.py with data_root=$DATA \
    num_gpus=1 num_nodes=1 per_gpu_batchsize=6 batch_size=6 \
    clip32_base224 text_t5_base image_size=224 vit_randaug \
    max_text_len=512 seed=42 \
    log_dir=$LOG precision=32 max_epoch=5 learning_rate=1.5e-4
```

## Inference

For evaluation, use the following commands:

**Configuration 1: Testing ResNet and BERT**  
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

**Configuration 2: Testing CLIP and T5**  
```bash
export DATA="/path/to/test/data"
export LOG="/path/to/logs"

CUDA_VISIBLE_DEVICES=0 python run.py with data_root=$DATA \
    num_gpus=1 num_nodes=1 per_gpu_batchsize=6 batch_size=6 test_only=True \
    clip32_base224 text_t5_base image_size=224 vit_randaug \
    log_dir=$LOG precision=32 max_text_len=512 \
    load_path="/path/to/checkpoints/model.ckpt"
```

## Highlights

- **Innovative Approach**: Simulates debates among LLMs to capture conflicting sarcastic rationales.
- **Enhanced Multimodal Understanding**: Combines textual and visual clues effectively for sarcasm detection.
- **Open-World Knowledge Utilization**: Integrates contextual information from diverse domains for robust sarcasm interpretation.
- **Comprehensive Framework**: Includes a debate module for generating sentiment rationales and a judge module for nuanced sentiment classification.

## Citation

If you use this framework in your research, please cite:

```
@misc{zhou2024ldgnet,
    title={LDGNet: LLMs Debate-Guided Network for Multimodal Sarcasm Detection},
    url={https://github.com/LDGNet-project/LDGNet},
    author={Zhou, Hengyang and Yan, Jinwu and Chen, Yaqing and Hong, Rongman and Zuo, Wenbo and Jin, Keyan},
    month={November},
    year={2024}
}
```

## Acknowledgements

This work builds on the strengths of leading multimodal frameworks and pushes the boundaries of sarcasm detection with unique debate-based methodologies.

---
