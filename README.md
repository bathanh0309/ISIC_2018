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
│   ├── models/            # Model checkpoints (.pt files)
│   ├── figures/           # Visualizations (confusion matrix, history)
│   └── submissions/       # Prediction CSVs
├── GroundTruth/           # Ground truth CSVs (không push lên git)
├── Input/                 # Ảnh training/val/test (không push lên git)
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```

![Dataset Overview](outputs/figures/dataset_overview.png)
![Class Distribution Comparison](outputs/figures/class_distribution_comparison.png)

---

## Chạy Training

### Bước 1: Mở notebook
Mở file `main.ipynb` trong Jupyter hoặc VS Code

### Bước 2: Chạy tuần tự các cells
```python
# Cell 1: Import modules và cấu hình
# Cell 2: Load và chuẩn bị dữ liệu
# Cell 3: Khởi tạo model
# Cell 4: Training loop
# Cell 5-8: Evaluation và visualization
```

### Kết quả sau training:
- Model checkpoint: `outputs/models/efficientnet_b1_isic2018.pt`
- Confusion matrices: `outputs/figures/val_confusion_matrix.png`
- Training history: `outputs/figures/training_history.png`
- Predictions: `outputs/submissions/test_predictions.csv`

---

## Model Architecture

**EfficientNet-B1**
- **Input size**: 224×224 pixels
- **Parameters**: ~6.5M (trainable)
- **Pretrained**: ImageNet weights
- **Output**: 7 classes (skin lesion types)


---

## Training Configuration

### Hyperparameters chính:

| Parameter | Value | Mô tả |
|-----------|-------|-------|
| **Model** | EfficientNet-B1 | Pretrained trên ImageNet |
| **Input Size** | 224×224 | Reduced từ 240 để train nhanh hơn |
| **Batch Size** | 16 (CPU) / 64 (GPU) | Tối ưu cho CPU training |
| **Learning Rate** | 1e-4 (0.0001) | AdamW optimizer |
| **Weight Decay** | 1e-4 | L2 regularization |
| **Epochs** | 10 | Giảm từ 15 để train nhanh |
| **Validation** | Every 3 epochs | Giảm overhead |
| **Early Stopping** | Patience = 2 | Dừng sớm nếu không cải thiện |

### Loss Function
**CrossEntropyLoss**
- Standard loss cho multi-class classification
- Tính softmax probability cho 7 classes
- Không sử dụng label smoothing (để train nhanh hơn)

### Optimizer
**AdamW (Adam with Weight Decay)**
- Adaptive learning rate cho từng parameter
- Weight decay = 1e-4 để tránh overfitting
- Beta1 = 0.9, Beta2 = 0.999 (PyTorch defaults)

### Learning Rate Scheduler
**CosineAnnealingLR**
- Giảm learning rate theo hàm cosine
- T_max = 10 epochs
- Eta_min = 1e-6 (LR tối thiểu)
- Giúp model converge tốt hơn cuối training

### Data Augmentation (Training)
- **Resize**: 224 → 258 pixels
- **RandomResizedCrop**: 224×224, scale=(0.8, 1.0)
- **RandomHorizontalFlip**: p=0.5
- **RandomVerticalFlip**: p=0.5
- **RandomRotation**: ±20 degrees
- **ColorJitter**: brightness, contrast, saturation, hue
- **Normalization**: ImageNet mean/std

### Class Imbalance Handling
**WeightedRandomSampler**
- Dataset có imbalance ratio: ~58:1 (NV vs DF)
- Sử dụng weighted sampling để balance classes
- Đảm bảo mỗi class được sample đều trong training

---

## Training Outputs

### Model Checkpoint (`.pt` file)
Chứa:
- Model weights (`model_state_dict`)
- Optimizer state (`optimizer_state_dict`)
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

## Dataset

**ISIC 2018 Task 3: Lesion Diagnosis**
- **Training**: 10,015 images
- **Validation**: 193 images  
- **Test**: 1,512 images

**Class Distribution** (highly imbalanced):
- NV (Nevi): ~67% - Đa số
- MEL (Melanoma): ~11%
- BKL: ~11%
- BCC: ~5%
- AKIEC: ~3%
- DF: ~1% - Ít nhất
- VASC: ~1.5%

**Imbalance Ratio**: 58.3:1 (max:min)  
→ **Sử dụng WeightedRandomSampler** để cân bằng

---

## File Quan Trọng

### Configuration
- `scr/config.py` - Tất cả hyperparameters và paths

### Training
- `main.ipynb` - Notebook chính
- `scr/train.py` - Training loop utilities
- `scr/model.py` - Model architecture

### Data
- `scr/data_processing.py` - Load và parse data
- `scr/dataset.py` - PyTorch Dataset
- `scr/transforms.py` - Data augmentation

### Checkpoints
- `outputs/models/efficientnet_b1_isic2018.pt` - Best model

---
