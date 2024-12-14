# BarcodeMamba

BarcodeMamba, a performant and efficient foundation model for DNA barcodes in biodiversity analysis.

- Check out our [paper](https://openreview.net/forum?id=6ohFEFTr10)
- Check out our [poster](https://huggingface.co/bioscan-ml/BarcodeMamba/resolve/main/poster/BarcodeMamba_Poster.pdf)
- Our pretrained models are at [HuggingFace](https://huggingface.co/bioscan-ml/BarcodeMamba)

![poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202024/105938.png)

## Reproducing the results

These instructions are for Linux (or Windows Subsystem for Linux).

1. Install CUDA 12.1 from the [NVIDIA CUDA Toolkit website](https://developer.nvidia.com/cuda-downloads), or load the cuda-12.1 module if it has been installed.

```bash
source /etc/profile.d/lmod.sh
module load cuda-12.1
```

2. Install the required libraries:

```bash
git clone https://github.com/bioscan-ml/BarcodeMamba.git
cd BarcodeMamba
conda env create -n BarcodeMamba -f environment.yml
conda activate BarcodeMamba
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.1/mamba_ssm-2.2.1+cu122torch2.3cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.3.0.post1/causal_conv1d-1.3.0.post1+cu122torch2.3cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

3. Download the datasets preprocessed by [BarcodeBERT](https://github.com/bioscan-ml/BarcodeBERT/) authors

```bash
mkdir data
wget https://vault.cs.uwaterloo.ca/s/x7gXQKnmRX3GAZm/download -O data.zip
unzip data.zip
mv new_data/* data/
rm -r new_data
rm data.zip
```

4. We recommend using A40 GPUs to avoid [IndexError](https://github.com/state-spaces/mamba/issues/361)


### Pretraining BarcodeMamba from scratch

- For pretraining with character-level tokenization:

```bash
python train.py dataset=CanadianInvertebrates1.5M-pretrain tokenizer=char dataset.input_path=/absolute/path/to/data
```

If you are using multiple gpus, use torchrun to start training:

```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    train.py dataset=CanadianInvertebrates1.5M-pretrain tokenizer=char dataset.input_path=/absolute/path/to/data
```

- Pretraining with K-mer tokenization:

```bash
python train.py dataset=CanadianInvertebrates1.5M-pretrain tokenizer=k-mer tokenizer.k_mer=6 dataset.input_path=/absolute/path/to/data
```

### Finetuning a pretrained BarcodeMamba

- Finetuning and testing with character-level tokenization:

```bash
python train.py dataset=CanadianInvertebrates1.5M-finetune dataset.input_path=/absolute/path/to/data model.n_classes=1653 tokenizer=char train.pretrained_model_path=/absolute/path/to/ckpt
```

- Finetuning and testing with K-mer tokenization:

```bash
python train.py dataset=CanadianInvertebrates1.5M-finetune dataset.input_path=/absolute/path/to/data model.n_classes=1653 tokenizer=k_mer tokenizer.k_mer=6 train.pretrained_model_path=/absolute/path/to/ckpt
```

### Using BarcodeMamba pretrained models for probing
Run the following commands to build a folder_of_pretrained_barcode_mamba: (taking [`BarcodeMamba-dim384-layer2-char`](https://huggingface.co/bioscan-ml/BarcodeMamba/tree/main/BarcodeMamba-dim384-layer2-char) as an example)

```
mkdir BarcodeMamba-dim384-layer2-char && cd BarcodeMamba-dim384-layer2-char
mkdir .hydra checkpoints
wget -O .hydra/config.yaml https://huggingface.co/bioscan-ml/BarcodeMamba/resolve/main/BarcodeMamba-dim384-layer2-char/.hydra/config.yaml?download=true
wget -O checkpoints/last.ckpt https://huggingface.co/bioscan-ml/BarcodeMamba/resolve/main/BarcodeMamba-dim384-layer2-char/checkpoints/last.ckpt?download=true
```

- Linear probing on seen species:

Note that this will generate a `probing_outputs` directory for logs.

```bash
python linear_probing.py -d /absolute/path/to/folder_of_pretrained_barcode_mamba --input-path /absolute/path/to/data 
```

- 1-NN probing on unseen species:

Note that this will generate a `probing_outputs` directory for logs.

```bash
python knn_probing.py -d /absolute/path/to/folder_of_pretrained_barcode_mamba --input-path /absolute/path/to/data 
```

### Baselines
For BarcodeBERT, CNN, DNABERT, DNABERT-2 baselines, please refer to the instructions in [BarcodeBERT](https://github.com/bioscan-ml/BarcodeBERT/).

- Finetuning a HyenaDNA model on our task.

```bash
python ssm_baselines.py model_name=hyenadna dataset.input_path=/absolute/path/to/data checkpoint=LongSafari/hyenadna-small-32k-seqlen-hf
```

- Finetuning a Caduceus model on our task.

```bash
python ssm_baselines.py model_name=mambadna dataset.input_path=/absolute/path/to/data checkpoint=kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3
```

- Using pretrained baseline models for probing

Linear probing:

Note that this will generate a `baseline_probing_outputs_linear` directory for logs.

```bash
python linear_probing_baseline.py --model-name hyenadna --input_path /absolute/path/to/data --checkpoint LongSafari/hyenadna-small-32k-seqlen-hf
python linear_probing_baseline.py --model-name mambadna --input_path /absolute/path/to/data --checkpoint kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3
```

1-NN probing:

Note that this will generate a `baseline_probing_outputs_knn` directory for logs.

```bash
python knn_probing_baseline.py --model-name hyenadna --input_path /absolute/path/to/data --checkpoint LongSafari/hyenadna-small-32k-seqlen-hf
python knn_probing_baseline.py --model-name mambadna --input_path /absolute/path/to/data --checkpoint kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3
```

Other variants of HyenaDNA and Caduceus models used in our paper:
```bash
checkpoint=LongSafari/hyenadna-tiny-1k-seqlen-d256-hf
checkpoint=kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3
checkpoint=kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16
checkpoint=kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16
```

## Citation

If you find BarcodeMamba useful, please consider citing:
```
@inproceedings{
gao2024barcodemamba,
title={BarcodeMamba: State Space Models for Biodiversity Analysis},
author={Tiancheng Gao and Graham W.~Taylor},
booktitle={{NeurIPS} 2024 Workshop on Foundation Models for Science: Progress, Opportunities, and Challenges},
year={2024},
url={https://openreview.net/forum?id=6ohFEFTr10}
}
```
