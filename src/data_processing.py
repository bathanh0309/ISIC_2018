
import os
import pandas as pd
from typing import Tuple, Dict


def parse_ground_truth(df_raw: pd.DataFrame, img_dir: str, split_name: str = "train") -> pd.DataFrame:

    df = df_raw.copy()
    
    # Detect image_id column (first column or contains 'image')
    img_col = df.columns[0]
    
    # Detect label columns (one-hot encoded or single label column)
    # ISIC2018 Task 3 uses one-hot encoding for 7 disease categories
    potential_label_cols = [col for col in df.columns if col != img_col]
    
    # Check if one-hot encoded (multiple binary columns)
    if len(potential_label_cols) > 1 and df[potential_label_cols].isin([0, 1]).all().all():
        # One-hot encoded: find the column with 1
        df['label'] = df[potential_label_cols].idxmax(axis=1)
    else:
        # Single label column
        df['label'] = df[potential_label_cols[0]]
    
    # Create image_id and image_path
    df['image_id'] = df[img_col].astype(str)
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(img_dir, f"{x}.jpg"))
    
    # Select relevant columns
    df_clean = df[['image_id', 'label', 'image_path']].copy()
    
    # Print statistics
    print(f"\n=== {split_name.upper()} DATASET ===")
    print(f"Total samples: {len(df_clean)}")
    print(f"\nLabel distribution:")
    label_counts = df_clean['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = 100 * count / len(df_clean)
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    return df_clean


def create_label_mapping(df_train: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str], int]:

    all_labels = sorted(df_train['label'].unique())
    label2idx = {label: idx for idx, label in enumerate(all_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    num_classes = len(all_labels)
    
    print(f"\n=== LABEL MAPPING ===")
    print(f"Number of classes: {num_classes}")
    for label, idx in label2idx.items():
        print(f"  {idx}: {label}")
    
    return label2idx, idx2label, num_classes


def apply_label_mapping(df: pd.DataFrame, label2idx: Dict[str, int]) -> pd.DataFrame:

    df['label_idx'] = df['label'].map(label2idx)
    return df


def check_class_imbalance(df_train: pd.DataFrame, threshold: float = 3.0) -> Tuple[bool, float]:

    train_label_counts = df_train['label_idx'].value_counts().sort_index()
    max_count = train_label_counts.max()
    min_count = train_label_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\n=== CLASS IMBALANCE ANALYSIS ===")
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
    print(f"Max class count: {max_count}")
    print(f"Min class count: {min_count}")
    
    use_weighted_sampler = imbalance_ratio > threshold
    
    if use_weighted_sampler:
        print(f"High class imbalance detected (ratio > {threshold}).")
        print("   Will use WeightedRandomSampler for balanced training.")
    else:
        print(f"Class distribution is balanced (ratio <= {threshold}).")
        print("   Will use standard random sampling.")
    
    return use_weighted_sampler, imbalance_ratio


def load_all_data(train_csv: str, val_csv: str, test_csv: str,
                  train_img_dir: str, val_img_dir: str, test_img_dir: str) -> Tuple:

    # Read CSVs
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    df_train_raw = pd.read_csv(train_csv)
    df_val_raw = pd.read_csv(val_csv)
    df_test_raw = pd.read_csv(test_csv)
    
    # Parse ground truth
    df_train = parse_ground_truth(df_train_raw, train_img_dir, "train")
    df_val = parse_ground_truth(df_val_raw, val_img_dir, "val")
    df_test = parse_ground_truth(df_test_raw, test_img_dir, "test")
    
    # Create label mapping
    label2idx, idx2label, num_classes = create_label_mapping(df_train)
    
    # Apply label mapping
    df_train = apply_label_mapping(df_train, label2idx)
    df_val = apply_label_mapping(df_val, label2idx)
    df_test = apply_label_mapping(df_test, label2idx)
    
    # Check class imbalance
    use_weighted_sampler, _ = check_class_imbalance(df_train)
    
    print("\n" + "=" * 60)
    print("DATA LOADING COMPLETE")
    print("=" * 60)
    
    return df_train, df_val, df_test, label2idx, idx2label, num_classes, use_weighted_sampler


if __name__ == "__main__":
    # Test data loading
    from config import PATH_TRAIN_CSV, PATH_VAL_CSV, PATH_TEST_CSV
    from config import DIR_TRAIN_IMG, DIR_VAL_IMG, DIR_TEST_IMG
    
    df_train, df_val, df_test, label2idx, idx2label, num_classes, use_weighted = load_all_data(
        PATH_TRAIN_CSV, PATH_VAL_CSV, PATH_TEST_CSV,
        DIR_TRAIN_IMG, DIR_VAL_IMG, DIR_TEST_IMG
    )
    
    print(f"\nDataframes loaded successfully!")
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
