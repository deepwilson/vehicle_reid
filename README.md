# Readme 
# Vehicle Re-Identification Framework

This repository contains the code and resources for training, testing, and evaluating vehicle re-identification models using deep learning techniques. The directory structure and main components of this framework are detailed below.

---

## Directory Structure

```
.
├── .github/
├── docs/
├── logs/
│   └── resnet50-veri/
│       └── .gitkeep
├── src/
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── center_loss.py
│   │   ├── cross_entropy_loss.py
│   │   ├── hard_mine_triplet_loss.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mobilenet.py
│   │   ├── resnet.py
│   │   ├── resnet18.py
│   │   ├── resnet18_cbam.py
│   │   ├── resnet18_cbam_concatfusion.py
│   │   ├── resnet18_fusion.py
│   │   ├── samobilenet.py
│   │   ├── seresnet.py
│   │   ├── seresnet18.py
│   │   ├── seresnet18_additionfusion.py
│   │   ├── seresnet18_concatfusion.py
│   │   ├── seresnet18_test_1.py
│   │   ├── seresnet18fusion.py
│   ├── utils/
│       ├── __init__.py
│       ├── custom_losses.py
│       ├── custom_models.py
│       ├── data_manager.py
│       ├── dataset_loader.py
│       ├── eval_metrics.py
│       ├── lr_schedulers.py
│       ├── optimizers.py
│       ├── samplers.py
│       ├── transforms.py
├── .gitignore
├── LICENSE
├── README.md
├── TODO.md
├── args.py
├── main.py
├── mixup_alternate_rows.py
├── output_image.jpg
├── resnet50-19c8e357.pth
├── test.sh
├── train.sh
```

---

## Requirements

- Python 3.8+
- Required libraries:
  - TensorFlow / PyTorch
  - NumPy
  - Matplotlib
  - Additional dependencies (listed in `requirements.txt`)

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd vehicle_reid
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Directory Descriptions

- **`src/losses/`**: Contains implementations of custom loss functions used in training models.
  - `center_loss.py`: Center loss implementation.
  - `cross_entropy_loss.py`: Cross-entropy loss.
  - `hard_mine_triplet_loss.py`: Hard-mining triplet loss.

- **`src/models/`**: Defines model architectures for vehicle re-identification.
  - `resnet18.py`: ResNet-18 base architecture.
  - `seresnet18.py`: SE-ResNet-18 variant.
  - `*_fusion.py`: Various fusion-based models.

- **`src/utils/`**: Utility scripts for data loading, augmentation, optimizers, and evaluation.
  - `dataset_loader.py`: Handles data loading and preprocessing.
  - `eval_metrics.py`: Metrics for model evaluation.
  - `custom_models.py`: Extensions or custom modifications of standard models.

- **`logs/`**: Stores logs and checkpoints from training.

- **`docs/`**: Documentation for the project.

---

## Usage

### Training

To train a model:
```bash
bash train.sh
```

### Evaluation

To test the model:
```bash
python3 main.py --data_path <data-path> --resume <checkpoint-path>
```

### Visualizing Logs

To visualize training logs using TensorBoard:
```bash
python3 -m tensorboard.main --logdir logs/
```

---


## Notes

- All checkpoints are stored in the `logs/` directory.
- Fusion models (`*_fusion.py`) implement advanced techniques for feature aggregation.

---
## Report
Report can be found at https://github.com/deepwilson/vehicle_reid/blob/main/EEEM071_Coursework_Report.pdf
## Feedback on research (Course module leader)

- The research work received the following feedback from the professor: 
![image](https://github.com/user-attachments/assets/2e0204e8-1a4e-4a74-b0fd-4d4098db9f8b)

