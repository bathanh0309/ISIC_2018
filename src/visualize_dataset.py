"""
Dataset Visualization for ISIC 2018
Trực quan hóa phân phối dataset train/val/test
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def plot_dataset_overview(df_train, df_val, df_test, save_path=None):
    """
    Vẽ tổng quan về dataset: số lượng samples và class distribution
    
    Args:
        df_train: DataFrame training data
        df_val: DataFrame validation data
        df_test: DataFrame test data
        save_path: Đường dẫn lưu figure (optional)
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure với 4 subplots (2x2)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.01)
    
    # === 1. Số lượng samples trong mỗi split (Bar chart) ===
    ax1 = fig.add_subplot(gs[0, 0])
    splits = ['Train', 'Val', 'Test']
    counts = [len(df_train), len(df_val), len(df_test)]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax1.bar(splits, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('Dataset Split Distribution', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3)
    
    # Thêm giá trị trên mỗi cột
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # === 2. Tỷ lệ phần trăm (Pie chart) ===
    ax2 = fig.add_subplot(gs[0, 1])
    total = sum(counts)
    
    wedges, texts, autotexts = ax2.pie(counts, labels=splits, autopct='%1.1f%%',
                                         colors=colors, startangle=90,
                                         radius=1.25,
                                         textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Dataset Split Percentage', fontsize=14, fontweight='bold', pad=15)
    
    # === 3. Training set class distribution (Horizontal bar) ===
    train_class_counts = df_train['label'].value_counts().sort_index()
    
    ax3 = fig.add_subplot(gs[1, 0])
    
    train_sorted = train_class_counts.sort_values(ascending=True)
    
    # Custom color map as requested
    class_color_map = {
        'NV': 'purple',
        'MEL': 'darkgreen',
        'BKL': 'lightgreen',
        'BCC': '#FFFACD',  # LightYellow
        'AKIEC': 'gold',   # Vàng đậm
        'VASC': 'orange',
        'DF': 'red'
    }
    
    # Fallback for unexpected classes if any
    unique_classes = sorted(train_class_counts.index)
    for label in unique_classes:
        if label not in class_color_map:
             class_color_map[label] = 'gray'

    colors_bar = [class_color_map[label] for label in train_sorted.index]
    
    bars = ax3.barh(range(len(train_sorted)), train_sorted.values, 
                     color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_yticks(range(len(train_sorted)))
    ax3.set_yticklabels(train_sorted.index)
    ax3.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
    ax3.set_title('Training Set - Class Distribution', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(axis='x', alpha=0.3)
    
    # Thêm giá trị
    for i, (bar, count) in enumerate(zip(bars, train_sorted.values)):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2.,
                f' {count:,} ({count/len(df_train)*100:.1f}%)',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # === 4. Class imbalance visualization (Pie chart) ===
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Sort by count descending
    train_sorted_desc = train_class_counts.sort_values(ascending=False)
    
    # Use consistent colors
    colors_pie = [class_color_map[label] for label in train_sorted_desc.index]
    
    wedges, texts = ax4.pie(train_sorted_desc.values, 
                                         labels=train_sorted_desc.index,
                                         colors=colors_pie,
                                         startangle=90,
                                         radius=1.2,
                                         textprops={'fontsize': 9})
    
    ax4.set_title('Training Set - Class Imbalance', fontsize=14, fontweight='bold', pad=15)
    
    # Main title
    fig.suptitle('ISIC 2018 Dataset Overview', fontsize=18, fontweight='bold', y=0.98)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    
    plt.tight_layout()
    return fig


def plot_class_distribution_comparison(df_train, df_val, df_test, save_path=None):
    """
    So sánh class distribution giữa train/val/test bằng stacked bar chart
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get all unique labels
    all_labels = sorted(set(list(df_train['label'].unique()) + 
                           list(df_val['label'].unique()) + 
                           list(df_test['label'].unique())))
    
    # Count for each split
    train_counts = [df_train[df_train['label'] == label].shape[0] for label in all_labels]
    val_counts = [df_val[df_val['label'] == label].shape[0] for label in all_labels]
    test_counts = [df_test[df_test['label'] == label].shape[0] for label in all_labels]
    
    x = np.arange(len(all_labels))
    width = 0.6
    
    # Stacked bars
    p1 = ax.bar(x, train_counts, width, label='Train', color='#FF6B6B', alpha=0.8)
    p2 = ax.bar(x, val_counts, width, bottom=train_counts, label='Val', color='#4ECDC4', alpha=0.8)
    p3 = ax.bar(x, test_counts, width, 
                bottom=np.array(train_counts) + np.array(val_counts),
                label='Test', color='#45B7D1', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Stacked Class Distribution (Train + Val + Test)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test visualization
    from config import PATH_TRAIN_CSV, PATH_VAL_CSV, PATH_TEST_CSV
    from config import DIR_TRAIN_IMG, DIR_VAL_IMG, DIR_TEST_IMG, DIR_FIGURES
    from data_processing import load_all_data
    
    print("Loading data...")
    df_train, df_val, df_test, label2idx, idx2label, num_classes, use_weighted = load_all_data(
        PATH_TRAIN_CSV, PATH_VAL_CSV, PATH_TEST_CSV,
        DIR_TRAIN_IMG, DIR_VAL_IMG, DIR_TEST_IMG
    )
    
    print("\nGenerating visualizations...")
    
    # Main overview
    fig1 = plot_dataset_overview(df_train, df_val, df_test, 
                                  save_path=f"{DIR_FIGURES}/dataset_overview.png")
    
    # Comparison
    fig2 = plot_class_distribution_comparison(df_train, df_val, df_test,
                                              save_path=f"{DIR_FIGURES}/class_distribution_comparison.png")
    
    plt.show()
    print("\n✅ Visualization complete!")
