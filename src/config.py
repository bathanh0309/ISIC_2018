
import os
import random
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
# DATA PATHS
# ========================
# Ground Truth CSV files
PATH_TRAIN_CSV = r"D:\ISIC2018\GroundTruth\Training_GrounTruth\ISIC2018_Task3_Training_GroundTruth.csv"
PATH_VAL_CSV = r"D:\ISIC2018\GroundTruth\Validation_GroundTruth\ISIC2018_Task3_Validation_GroundTruth.csv"
PATH_TEST_CSV = r"D:\ISIC2018\GroundTruth\Test_GroundTruth\ISIC2018_Task3_Test_GroundTruth.csv"

# Image directories
DIR_TRAIN_IMG = r"D:\ISIC2018\Input\Training_Input"
DIR_VAL_IMG = r"D:\ISIC2018\Input\Validation_Input"
DIR_TEST_IMG = r"D:\ISIC2018\Input\Test_Input"

# Lesion groupings (optional)
PATH_LESION_GROUPINGS = r"D:\ISIC2018\Training_LesionGroupings.csv"


# ========================
# OUTPUT PATHS
# ========================
DIR_OUTPUT = "outputs"
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
    print(f"\nModel Path: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
