import os
import copy
from datetime import datetime
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as tsfm
import argparse


def parse_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tuning CLIP-based models.')
    parser.add_argument('--model_type', choices=['clip', 'medclip', 'biomedclip'], required=True,
                        help='Model backbone: clip, medclip, or biomedclip')
    parser.add_argument('--variant', type=str, required=True,
                        help='Model variant: e.g., B32, B16, RN50x4 for clip; vit, rn for medclip; vit for biomedclip')
    parser.add_argument('--mode', choices=['lp', 'mlp', 'ft', 'lora', 'zero-shot'], default='lp',
                        help='Fine-tuning method: lp, mlp, ft, lora, zero-shot')
    parser.add_argument('--seed', type=int, default=77, help='Random seed')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--path', type=str, default='./output/pred',
                        help='Path for logs and checkpoints')

    args = parser.parse_args()

    # Validate variant based on model_type
    valid_variants = {
        'clip': ['B32', 'B16', 'RN50x4'],
        'medclip': ['vit', 'rn'],
        'biomedclip': ['vit']
    }
    if args.variant not in valid_variants[args.model_type]:
        raise ValueError(f"Invalid variant '{args.variant}' for model_type '{args.model_type}'. "
                         f"Valid options: {valid_variants[args.model_type]}")

    return args


def make_true_labels(cxr_true_labels_path: str, cutlabels: bool = True) -> np.ndarray:
    """Load true binary labels from CSV for evaluation."""
    full_labels = pd.read_csv(cxr_true_labels_path)
    cxr_labels = get_categories_list(cxr_true_labels_path)
    if cutlabels:
        full_labels = full_labels.loc[:, cxr_labels]
    else:
        full_labels.drop(full_labels.columns[0], axis=1, inplace=True)
    return full_labels.to_numpy()


def get_categories_list(csv_file_path: str) -> list:
    """Get list of column names from CSV header."""
    df = pd.read_csv(csv_file_path, nrows=0)
    return df.columns.tolist()[7:]


def gettime_str() -> str:
    """Get current time as a formatted string."""
    return datetime.now().strftime("%Y%m%d%H")


class AvgMeter:
    """Compute and store the average, sum, and count of values."""

    def __init__(self, name: str = "Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val: float, count: int = 1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


def get_lr(optimizer) -> float:
    """Get the current learning rate from the optimizer."""
    return optimizer.param_groups[0]["lr"]


def get_n_params(model) -> int:
    """Get the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def create_directory_if_not_exists(directory_path: str):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def create_train_transform(val_transform: tsfm.Compose) -> tsfm.Compose:
    """Create training transform with augmentations."""
    train_transform_list = copy.deepcopy(val_transform.transforms)
    train_transform_list.insert(2, tsfm.RandomVerticalFlip(p=0.3))
    train_transform_list.insert(3, tsfm.RandomHorizontalFlip(p=0.3))
    train_transform_list.insert(4, tsfm.RandomRotation(degrees=15))
    train_transform_list.insert(5, tsfm.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    return tsfm.Compose(train_transform_list)


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def get_weight_decay_params(model):
    """Separate parameters into those with and without weight decay."""
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
    print("p_wd:", len(p_wd))
    print("p_non_wd:", len(p_non_wd))
