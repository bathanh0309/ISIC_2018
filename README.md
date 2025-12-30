# ISIC 2018 Skin Lesion Classification

**Thành viên: Nguyễn Bá Thành**

Dự án classification các tổn thương da sử dụng EfficientNet-B1 trên dataset ISIC 2018.

---

## Cấu trúc Project

```
ISIC2018/
├── scr/                   # Source code modules
│   ├── __init__.py        # Package initializer
│   ├── config.py          # Cấu hình và hyperparameters
│   ├── data_processing.py # Xử lý dữ liệu và label mapping
│   ├── dataset.py         # PyTorch Dataset class
│   ├── transforms.py      # Data augmentation
│   ├── model.py           # Model architecture (EfficientNet-B1)
│   ├── train.py           # Training utilities
│   └── evaluate.py        # Evaluation và metrics
├── main.ipynb             # Notebook chính (đã được refactor)
├── outputs/               # Thư mục outputs
├── GroundTruth/           # Ground truth CSVs (không push lên git)
├── Input/                 # Ảnh training/val/test (không push lên git)
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```

![Dataset Overview](outputs/figures/dataset_overview.png)
![Class Distribution Comparison](outputs/figures/class_distribution_comparison.png)

---

## Training Outputs

Model mới train được lưu tại:
`outputs/models/efficientnet_b1_isic2018.pt`

### Model Checkpoint (.pt file)
Chứa:
- Model weights (model_state_dict)
- Optimizer state (optimizer_state_dict)
- Training history (loss, accuracy, F1 score)
- Best validation F1 score
- Label mappings (label2idx, idx2label)
- Epoch information

### Metrics được track:
- **Loss**: Training và validation loss
- **Accuracy**: Overall accuracy
- **Macro F1 Score**: F1 trung bình của 7 classes
- **Balanced Accuracy**: Accuracy có weight theo class
- **Learning Rate**: LR qua các epochs

---

## Training Configuration

| Parameter | Value | Mô tả |
|-----------|-------|-------|
| **Loss Function** | CrossEntropyLoss | Standard classification |
| **Optimizer** | AdamW | Adaptive LR weight decay (1e-4) |
| **Learning Rate** | 1e-4 | 0.0001 initial LR |
| **Data Augmentation** | Resize, Crop, Flip, Rotation, Normalize | Tăng cường dữ liệu |
| **Class Imbalance** | WeightedRandomSampler | Cân bằng tỉ lệ các class (ratio ~58:1) |


---

## Model Architecture

**EfficientNet-B1**
- **Input size**: 224×224 pixels
- **Parameters**: ~6.5M (trainable)
- **Pretrained**: ImageNet weights
- **Output**: 7 classes (skin lesion types)

---

## Training History & Evaluation

![Training History](outputs/figures/training_history.png)
![Validation Confusion Matrix](outputs/figures/val_confusion_matrix.png)
![Test Confusion Matrix](outputs/figures/test_confusion_matrix.png)
