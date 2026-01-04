"""
Configuration for ISIC 2018 Task 3 Classification.
Supports both Windows local and Codespaces (Linux) environments.
"""

import os
import random
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch

# ========================
# RANDOM SEED
# ========================
SEED = 42

def set_seed(seed=SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========================
# DEVICE CONFIGURATION
# ========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========================
# MODEL CONFIGURATION
# ========================
MODEL_NAME = 'efficientnet_b1'  # Changed from B3 to B1
IMG_SIZE = 224  # Reduced from 240 to 224 for faster processing
NUM_CLASSES = 7  # ISIC 2018 has 7 disease categories

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========================
# TRAINING HYPERPARAMETERS
# ========================
# Optimizer
LEARNING_RATE = 1e-4
# WEIGHT_DECAY = 1e-4  # Old value
WEIGHT_DECAY = 1e-3  # Increased to reduce overfitting

# Dropout (New parameters)
DROP_RATE = 0.3       # Dropout rate for classifier 
DROP_PATH_RATE = 0.2  # Stochastic depth rate

# Training
# NUM_EPOCHS = 20  # Old value
NUM_EPOCHS = 5     # Train for 5 additional epochs
BATCH_SIZE = 64 if torch.cuda.is_available() else 64
NUM_WORKERS = 0
VAL_EVERY_N_EPOCHS = 1  # Validate every epoch for fine-tuning

# Loss function
# USE_LABEL_SMOOTHING = False # Old value
USE_LABEL_SMOOTHING = True    # Enabled to reduce overfitting
LABEL_SMOOTHING = 0.1

# Test-Time Augmentation (TTA)
USE_TTA_VALIDATION = False  # Set True to use TTA during val (slower)
USE_TTA_TEST = True         # Set True for final test results (better metrics)

# Learning rate scheduler
USE_COSINE_SCHEDULER = True

# Early stopping
EARLY_STOP_PATIENCE = 3
MONITOR_METRIC = 'val_f1'
SAVE_EVERY_N_EPOCHS = 1


# ========================
# AUTO-DETECT DATA PATHS
# ========================
# Compute REPO_ROOT from this file's location
REPO_ROOT = Path(__file__).resolve().parents[1]

# DATA_ROOT can be overridden via environment variable
# Default: REPO_ROOT/data/ISIC2018
DATA_ROOT = Path(os.getenv("ISIC2018_DATA_ROOT", REPO_ROOT / "data" / "ISIC2018"))


def find_file(directory: Path, patterns: List[str]) -> Optional[Path]:
    """Find first file matching any of the patterns in directory (recursive)."""
    if not directory.exists():
        return None
    for pattern in patterns:
        matches = list(directory.rglob(pattern))
        if matches:
            return matches[0]
    return None


def find_dir_with_images(directory: Path, patterns: List[str], img_ext: str = "*.jpg") -> Optional[Path]:
    """Find first directory matching patterns that contains image files."""
    if not directory.exists():
        return None
    for pattern in patterns:
        # Try exact pattern match
        for subdir in directory.rglob(pattern):
            if subdir.is_dir() and list(subdir.glob(img_ext)):
                return subdir
    return None


# Try to find CSV files (Ground Truth)
_csv_patterns_train = ["*Task3*Training*GroundTruth*.csv", "*Training*Ground*.csv", "Training*.csv"]
_csv_patterns_val = ["*Task3*Validation*GroundTruth*.csv", "*Validation*Ground*.csv", "Validation*.csv"]
_csv_patterns_test = ["*Task3*Test*GroundTruth*.csv", "*Test*Ground*.csv", "Test*.csv"]

# Try to find image directories
_img_dir_patterns_train = ["*Training*Input*", "Training_Input", "*Task3*Training*Input*"]
_img_dir_patterns_val = ["*Validation*Input*", "Validation_Input", "*Task3*Validation*Input*"]
_img_dir_patterns_test = ["*Test*Input*", "Test_Input", "*Task3*Test*Input*"]


def _auto_detect_csv(patterns: List[str], fallback: str) -> str:
    """Auto-detect CSV path or return fallback."""
    found = find_file(DATA_ROOT, patterns)
    if found:
        return str(found)
    # Try REPO_ROOT as well (for backward compatibility)
    found = find_file(REPO_ROOT, patterns)
    if found:
        return str(found)
    return fallback


def _auto_detect_img_dir(patterns: List[str], fallback: str) -> str:
    """Auto-detect image directory or return fallback."""
    found = find_dir_with_images(DATA_ROOT, patterns)
    if found:
        return str(found)
    # Try REPO_ROOT as well
    found = find_dir_with_images(REPO_ROOT, patterns)
    if found:
        return str(found)
    return fallback


# ========================
# DATA PATHS (Auto-detected)
# ========================
PATH_TRAIN_CSV = _auto_detect_csv(
    _csv_patterns_train,
    str(DATA_ROOT / "Training_GroundTruth" / "ISIC2018_Task3_Training_GroundTruth.csv")
)
PATH_VAL_CSV = _auto_detect_csv(
    _csv_patterns_val,
    str(DATA_ROOT / "Validation_GroundTruth" / "ISIC2018_Task3_Validation_GroundTruth.csv")
)
PATH_TEST_CSV = _auto_detect_csv(
    _csv_patterns_test,
    str(DATA_ROOT / "Test_GroundTruth" / "ISIC2018_Task3_Test_GroundTruth.csv")
)

DIR_TRAIN_IMG = _auto_detect_img_dir(_img_dir_patterns_train, str(DATA_ROOT / "Training_Input"))
DIR_VAL_IMG = _auto_detect_img_dir(_img_dir_patterns_val, str(DATA_ROOT / "Validation_Input"))
DIR_TEST_IMG = _auto_detect_img_dir(_img_dir_patterns_test, str(DATA_ROOT / "Test_Input"))

# Lesion groupings (optional)
PATH_LESION_GROUPINGS = find_file(REPO_ROOT, ["*Lesion*Groupings*.csv"])
if PATH_LESION_GROUPINGS:
    PATH_LESION_GROUPINGS = str(PATH_LESION_GROUPINGS)


# ========================
# OUTPUT PATHS
# ========================
DIR_OUTPUT = str(REPO_ROOT / "outputs")
DIR_MODELS = os.path.join(DIR_OUTPUT, "models")
DIR_SUBMISSIONS = os.path.join(DIR_OUTPUT, "submissions")
DIR_FIGURES = os.path.join(DIR_OUTPUT, "figures")

# Model checkpoint path
MODEL_PATH = os.path.join(DIR_MODELS, f"{MODEL_NAME}_isic2018.pt")

# Create output directories
os.makedirs(DIR_MODELS, exist_ok=True)
os.makedirs(DIR_SUBMISSIONS, exist_ok=True)
os.makedirs(DIR_FIGURES, exist_ok=True)


# ========================
# LABEL MAPPING
# ========================
# Will be populated from data
LABEL2IDX = {}
IDX2LABEL = {}


def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Random Seed: {SEED}")
    print(f"\nModel: {MODEL_NAME}")
    print(f"Input Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"\nBatch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Early Stop Patience: {EARLY_STOP_PATIENCE}")
    print(f"\nREPO_ROOT: {REPO_ROOT}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"\nTrain CSV: {PATH_TRAIN_CSV}")
    print(f"Val CSV: {PATH_VAL_CSV}")
    print(f"Test CSV: {PATH_TEST_CSV}")
    print(f"\nTrain Images: {DIR_TRAIN_IMG}")
    print(f"Val Images: {DIR_VAL_IMG}")
    print(f"Test Images: {DIR_TEST_IMG}")
    print(f"\nModel Path: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
