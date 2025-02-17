<p align="center">
  <img src="data/image.png" alt="Project Logo" width="600"/>
</p>

<div align="center">

# Open Optical Character Recognition (OOCR)

[![Documentation](https://img.shields.io/github/actions/workflow/status/heoj1n/oocr/docs.yml?label=docs&color=22863a)](https://heoj1n.github.io/oocr/)
[![Python](https://img.shields.io/badge/python-3.10%2B-22863a?logo=python&logoColor=ffffff)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/heoj1n/oocr?color=22863a)](https://github.com/heoj1n/oocr/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/heoj1n/oocr?color=22863a)](https://github.com/heoj1n/oocr/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/heoj1n/oocr?color=22863a)](https://github.com/heoj1n/oocr/network/members)
[![GitHub Contributors](https://img.shields.io/github/contributors/heoj1n/oocr?color=22863a)](https://github.com/heoj1n/oocr/graphs/contributors)

**A Framework for Training Machine Learning Models for Character Recognition**

[Supported Datasets](#-supported-datasets) •
[Training Examples](#-training-examples) •
[Inference](#-inference) •
[Data Generation](#-data-generation)

</div>

OOCR is an open-source toolkit for training and deploying OCR models. It provides:
- **Easy-to-use CLI** for training and inference
- **Multiple Model Support**: TrOCR, Donut, and Nougat architectures
- **Data Processing**: Built-in support for IAM, MNIST, and SROIE datasets
- **Synthetic Data Generation**: Tools to create custom training data

Perfect for researchers and developers in search of a highly accurate OCR model for
handwritten and printed text recognition tasks.

# Development Setup (with uv)

Create virtual environment:
```bash
# Create and activate virtual environment
python -m venv .venv # with uv: uv venv
source .venv/bin/activate  # Unix/macOS
.venv\Scripts\activate  # Windows
```

Install dependencies:
```bash
# Install dependencies
uv pip install .  # main dependencies
uv pip install ".[dev]"  # with dev dependencies
uv pip install ".[docs]"  # with docs dependencies
uv pip install ".[dev,docs]"  # all dependencies

# Pre-commit hooks
uv pip install pre-commit
pre-commit install

# Apply formatting and linting
ruff check   # Lint all files in the current directory.
ruff format  # Format all files in the current directory.

# Run tests
pytest # all tests
pytest --cov=oocr/oocr # coverage
pytest tests/unit/test_models/test_trocr.py # specific test
```

# Supported Datasets
1. **IAM Words Dataset** - [Download Link](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database)
2. **Synthetic Dataset** - Romanian examples at [Download Link](https://drive.google.com/drive/folders/1ErvjszLBqVIrO7wnsVUc6zWv5CtPmgF_?usp=sharing) or create your own using the `generator.py` script. An alternative is to use the trdg library to generate synthetic data [Download Link](https://github.com/Belval/TextRecognitionDataGenerator), they use the same dataset structure.
3. **MNIST Dataset** - (Auto-downloads)
4. **SROIE Dataset** - (Auto-downloads with kaggle credentials) [Download Link](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database)

### Kaggle Setup:
You can to set up kaggle credentials to automatically download the SROIE, and IAM datasets.
Else you can download the dataset manually from the kaggle website.
Setup kaggle credentials:
1. First, create a Kaggle account if you don't have one at https://www.kaggle.com
2. Once logged in, go to your account settings:
- Click on your profile picture
- Go to "Settings"
- Scroll down to "API" section
- Click "Create New Token"
- This will download a file called kaggle.json
3. Create a directory in your home directory called `.kaggle`
```bash
# On Windows:
mkdir %USERPROFILE%\.kaggle

# On Linux/Mac:
mkdir ~/.kaggle
```
4. Move the kaggle.json file to the ~/.kaggle/ directory
- Ensure the file permissions are secure: chmod 600 ~/.kaggle/kaggle.json
3.Install kaggle if you haven't:
```bash
pip install kaggle
```


# Training

```sh
# Basic training
python train.py --data data/datasets/ocr_dataset --epochs 5

# Different datasets
python train.py --data data/datasets/iam_words --epochs 5 --dataset IAM
python train.py --epochs 5 --dataset MNIST --fraction 0.01  # Auto-downloads

# Alternative models
python train.py --data data/datasets/ocr_dataset --model naver-clova-ix/donut-base --batchsize 1
python train.py --data data/datasets/ocr_dataset --model facebook/nougat-base --batchsize 4

# Resume from checkpoint
python train.py --data data/datasets/ocr_dataset \
                --checkpoint data/output/ocr_dataset/trocr-large-handwritten/e20_lr1e-06_b4_1222/best_checkpoint.pt \
                --epochs 20

# Use config file
python train.py --config data/configs/ocr_config.yaml
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `data` | Training data directory path | Required |
| `--evaldata` | Evaluation data directory | `data/datasets/balcesu_test` |
| `--output` | Output directory for checkpoints/logs | `data/output` |
| `--checkpoint` | Path to resume training from | `None` |
| `--model` | Model choice (trocr-base/large, handwritten) | `microsoft/trocr-large-handwritten` |
| `--dataset` | Dataset type (IAM/MNIST/custom) | `custom` |
| `--epochs` | Number of training epochs | `5` |
| `--batchsize` | Training batch size | `4` |
| `--lr` | Learning rate | `1e-6` |
| `--lr_patience` | Epochs before LR reduction | `2` |
| `--val_iters` | Evaluation frequency (epochs) | `1` |
| `--num_samples` | Validation samples to display | `10` |

# Inference

```bash
# Basic usage with just an image
python inference.py --image_path path/to/image.jpg

# Using a specific model and checkpoint
python inference.py --image_path path/to/image.jpg --model_path path/to/model --checkpoint_path path/to/checkpoint.pt

# Using all options
python inference.py --image_path path/to/image.jpg \
                   --model_path custom/model/path \
                   --checkpoint_path custom/checkpoint.pt \
                   --reference_text_path custom/reference.txt \
                   --predictions_file custom/predictions.json \
                   --draw # Draw bounding boxes on the image

# One-line string:
python inference.py --image_path data\inference\Balcescu.png --checkpoint_path data\output\oscar_v2_50k\trocr-large-handwritten\e20_lr1e-06_b4_fr1.0_tfr1.0_balcesu_test\trocr-large-handwritten-best-char_acc.pt --predictions_file data\output\sroie\SROIE2019\trocr-base-printed\e10_lr1e-06_b8_fr1.0_tfr1.0_1\predictions\test_preds_e2.json
```

# 📊 Benchmarks

The TrOCR framework has been evaluated on multiple datasets, showing strong performance on both handwritten and historical document recognition tasks:

### Performance Overview

| Dataset | Model | WER (%) | CER (%) | Char Acc (%) | Test Loss |
|---------|--------|----------|----------|--------------|-----------|
| IAM Words | TrOCR-large | 12.33 | 5.32 | 92.61 | 0.0059 |
| SROIE | TrOCR-base-printed | 4.90 | 0.79 | 97.90 | 0.0030 |

### Dataset Details
- **IAM Words**: Standard benchmark for English handwritten text recognition
- **OSCAR**: Custom 5k sample dataset with mixed printed/handwritten text
- **SROIE**: Receipt OCR dataset with printed text, focusing on retail receipts and invoices

### Key Observations
- Strong performance on standard handwriting benchmarks (IAM)
- Robust handling of mixed printed/handwritten content (OSCAR v2)
- Acceptable performance on historical documents despite their complexity
- Character-level accuracy consistently above 84% across all datasets
- Lower loss on historical documents mainly due corroded samples in the synthetic dataset

# Data Generation

The `generator.py` script creates synthetic OCR training data. It requires a text dataset (.txt file) and a set of fonts and backgrounds. An alternative is to use the trdg library to generate synthetic data [Download Link](https://github.com/Belval/TextRecognitionDataGenerator), they use the same dataset structure. You may run into some issues setting it up, because the trdg package does not receive support anymore. But once set up the integrated generator has better augmentation than the generator offered here.

You can download two romanian language datasets (wiki-dump and OSCAR dataset) using this script:
```sh
python data/generation/download_text_dataset.py
```

Next, consider the following input structure:
```sh
generation/
├── assets
    ├── fonts # place your fonts here
    ├── backgrounds # place your backgrounds here
└── data/
    ├── input/
        ├── text-dataset.txt # arbitrary text dataset
    ├── output/
        # empty directory
├── generator.py
```

Run this command in your environment to generate synthetic dataset:
```sh
python generator.py data/generation/data/input/text-dataset.txt data/generation/data/output/ocr_dataset --image_count 50 --sentences_per_page 1 --max_length 100 --characters "ăâîșțĂÂÎȘȚ"
```
We get a new directoru in under *generation/data/output*:
```sh
├── output/
    ├── ocr_dataset/
      ├── labels.csv
      ├── labels.txt
      └── images/
          ├── image0.png
          ├── image1.png
          └── ...
```

labels.txt:
```sh
{image0.png}\t{text0}
{image1.png}\t{text1}
...
```

You can use the generated dataset for training your OCR model by chosing the `--data` argument to point to the generated dataset and `--dataset` to `custom`.

# Acknowledgments

- https://github.com/microsoft/unilm/tree/master/trocr
- https://github.com/Belval/TextRecognitionDataGenerator
- https://www.kaggle.com/datasets/urbikn/sroie-datasetv2/data
- https://github.com/naver-ai/donut
- https://github.com/facebookresearch/nougat