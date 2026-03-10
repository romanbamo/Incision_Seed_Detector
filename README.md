# Incision Seed Detector

This repository contains a Deep Learning model based on **EfficientNet-B1** designed to automatically detect the starting "seed" (x, y coordinates) for a **Region Growing** algorithm in surgical incision images.

## Project Overview
The goal is to automate the initialization of segmentation algorithms. Instead of manually picking a starting point, this CNN predicts the most likely center of a surgical incision to begin the region-growing process.

- **Architecture:** EfficientNet-B1 (Backbone) + Linear Regression Head.
- **Input:** RGB Image (resized to 240x240).
- **Output:** Normalized (x, y) coordinates of the incision seed.
- **Framework:** PyTorch.

### Training Strategy

The model follows a two-step training process:

- Phase 1 (Frozen Backbone): Only the final fully connected layer is trained using a high learning rate (1e-3).

- Phase 2 (Fine-Tuning): The entire network is unfrozen and trained with a lower learning rate (1e-5) to refine the weights for surgical feature extraction.

### Results

The model outputs the coordinates which are then de-normalized to the original image size.

- Red Dot 🔴: Predicted Seed.

- Green Dot 🟢 : Ground Truth.


## Repository Structure
* `src/model.py`: Model architecture definition.
* `src/dataset.py`: Custom PyTorch Dataset for medical images and coordinate labels.
* `src/train.py`: Two-phase training script (Transfer Learning + Fine-Tuning).
* `requirements.txt`: Necessary Python libraries.

## Getting Started

### Prerequisites
* Python 3.8+
* CUDA-enabled GPU (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/romanbamo/Incision-Seed-Detector.git
   cd Incision-Seed-Detector

2. Install dependences
   ```bash
   pip install -r requirements.txt

3. Pre-trained weights can be download from:
[EfficientNet-B1-Pre-trained-Weight](https://drive.google.com/file/d/1Z1YE09sp-ZmbWWWcLTJrEoBXPn-jY-MH/view?usp=drive_link)

or installed from:
```bash
pip install gdown
gdown 1Z1YE09sp-ZmbWWWcLTJrEoBXPn-jY-MH -O models/best_model.pth
