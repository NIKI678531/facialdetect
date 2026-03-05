# facialdetect

An end-to-end computer vision toolkit for feature extraction, binary classification, and batch image filtering.

This project is built for practical workflows: detect image attributes, score large photo sets, and automate filtering pipelines with reproducible scripts.

## Why This Project

- Production-style CV workflow: extract features -> train classifier -> run batch inference -> filter outputs
- SigLIP2-powered embeddings for full-image, person-level, and face-level signals
- Flexible scripts for data analysis, threshold tuning, and quality control
- Easy to adapt for moderation, profile-photo quality checks, and attribute mining

## Main Capabilities

- Feature extraction from:
  - full image (`all`)
  - person crop (`person`)
  - face crop (`face` / `face_only`)
- PyTorch training with custom MLP + residual blocks
- Threshold-based binary inference for high-volume datasets
- Utility scripts for CSV processing, image copying, renaming, and evaluation

## Repository Structure

```text
facialdetect/
├── app/                          # Embedding extraction and app-level logic
├── infrastructure/               # Model wrappers and image processing utils
├── model/                        # Backbone/network definitions
├── tool/                         # Data processing and evaluation scripts
├── train.py                      # Training entry point
├── predict_single_pic.py         # Single-image inference
├── predict_batch.py              # Batch inference
├── predict_batch_from_csv.py     # Batch inference with CSV
└── README.md
```

## Tech Stack

- Python 3.10+
- PyTorch / TorchVision
- Transformers (SigLIP2)
- OpenCV + Pillow
- NumPy / Pandas / scikit-learn
- Flask (service-oriented extension)

## Installation

```bash
git clone https://github.com/NIKI678531/facialdetect.git
cd facialdetect
pip install -U pip
pip install torch torchvision transformers opencv-python pillow numpy pandas tqdm flask requests scikit-learn
```

## Quick Start

### 1. Train a model

```bash
python train.py PictureHighClarity person
```

Parameters:

- `feature`: target label name (for example `PictureHighClarity`)
- `location`: embedding source (`all`, `person`, or `face`)

### 2. Run batch inference

```bash
python predict_batch.py PictureHighClarity person model_weight/isPictureHighClarity/your_model.pth /path/to/images 0.65 1 female
```

Arguments:

- `feature`
- `location`
- `model_path`
- `image_dir`
- `threshold`
- `flag` (`1` for positive selection, `0` for inverse selection)
- `gender` (used by current script variant)

### 3. Run single-image inference

```bash
python predict_single_pic.py PictureHighClarity person model_weight/isPictureHighClarity/your_model.pth /path/to/image.jpg 0.65
```

## Notes for New Machines

- Some scripts contain local absolute paths (for example `/Users/...`) and should be updated for your environment.
- Current code defaults to `mps` on Apple Silicon in several places.
  - Use `cpu` or `cuda` if `mps` is unavailable.
- Keep large datasets and model weights outside the repo when possible.

## Suggested GitHub Description

Practical CV toolkit for SigLIP2 feature extraction, PyTorch training, and scalable batch image filtering.

## Roadmap Ideas

- Add `requirements.txt` with pinned versions
- Replace hard-coded paths with environment-based config
- Provide Docker image and API deployment guide
- Add benchmark scripts and sample dataset manifest

## License

No license file is currently included.
If you plan to open-source publicly, add a license such as MIT or Apache-2.0.
