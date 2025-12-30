# ISIC 2018 Skin Lesion Classification

Dự án classification các tổn thương da sử dụng EfficientNet-B1 trên dataset ISIC 2018.

## 📁 Cấu trúc Project

```
ISIC2018/
├── config.py              # Cấu hình và hyperparameters
├── data_processing.py     # Xử lý dữ liệu và label mapping
├── dataset.py             # PyTorch Dataset class
├── transforms.py          # Data augmentation
├── model.py               # Model architecture (EfficientNet-B1)
├── train.py               # Training utilities
├── evaluate.py            # Evaluation và metrics
├── main.ipynb             # Notebook chính (đã được refactor)
├── outputs/               # Thư mục outputs
│   ├── models/            # Model checkpoints
│   ├── figures/           # Visualizations
│   └── submissions/       # Prediction CSVs
├── GroundTruth/           # Ground truth CSVs (không push lên git)
└── Input/                 # Ảnh training/val/test (không push lên git)
```

## 🔧 Cài đặt

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies chính:
- PyTorch
- timm (EfficientNet models)
- scikit-learn
- pandas
- matplotlib
- seaborn
- Pillow

## 🚀 Sử dụng

### 1. Chuẩn bị dữ liệu

Đảm bảo các folder sau tồn tại và chứa đúng dữ liệu:
- `GroundTruth/Training_GrounTruth/` - Training labels
- `GroundTruth/Validation_GroundTruth/` - Validation labels  
- `GroundTruth/Test_GroundTruth/` - Test labels
- `Input/Training_Input/` - Training images
- `Input/Validation_Input/` - Validation images
- `Input/Test_Input/` - Test images

### 2. Chạy training

Mở và chạy `main.ipynb` trong Jupyter hoặc VS Code:

```python
# Cell 1: Import modules
# Cell 2: Load data
# Cell 3: Initialize model
# Cell 4: Training loop
# Cell 5-8: Evaluation và visualization
```

### 3. Test các module riêng lẻ

```bash
# Test config
python config.py

# Test data processing
python data_processing.py

# Test model
python model.py

# Test training utilities
python train.py
```

## 📊 Model Architecture

**EfficientNet-B1**
- Input size: 240x240
- Parameters: ~6.5M (giảm 40% so với B3)
- Pretrained: ImageNet

### Thay đổi từ phiên bản trước:
- ✅ Đổi từ EfficientNet-**B3** → **B1**
- ✅ Image size: ~~300~~ → **240**
- ✅ Checkpoint: `efficientnet_b1_isic2018.pt`
- ⚡ Training nhanh hơn ~30-40%
- 💾 Sử dụng ít memory hơn

## 🎯 Training Configuration

Các hyperparameters chính trong `config.py`:

```python
MODEL_NAME = 'efficientnet_b1'
IMG_SIZE = 240
BATCH_SIZE = 16  # (8 nếu CPU)
LEARNING_RATE = 3e-4
NUM_EPOCHS = 15
EARLY_STOP_PATIENCE = 3
```

## 📈 Evaluation Metrics

- Accuracy
- Macro F1 Score
- Balanced Accuracy
- Confusion Matrix
- Per-class Precision/Recall

## 💾 Checkpoints

Model checkpoints được lưu tại `outputs/models/efficientnet_b1_isic2018.pt` và bao gồm:
- Model weights
- Optimizer state
- Training history
- Best validation F1
- Label mappings

## 📝 Outputs

Sau khi training, các file sau được tạo:

### Models
- `outputs/models/efficientnet_b1_isic2018.pt` - Best model checkpoint

### Figures
- `outputs/figures/val_confusion_matrix.png` - Validation confusion matrix
- `outputs/figures/test_confusion_matrix.png` - Test confusion matrix
- `outputs/figures/training_history.png` - Training curves
- `outputs/figures/inference_demo.png` - Sample predictions

### Submissions
- `outputs/submissions/test_predictions.csv` - Test predictions với probabilities

## 🔍 Module Details

### `config.py`
- Centralized configuration
- Device setup
- Paths và hyperparameters
- Seed cho reproducibility

### `data_processing.py`
- Parse ground truth CSVs
- Tạo label mappings
- Phân tích class imbalance
- Load tất cả datasets

### `dataset.py`
- PyTorch Dataset class `ISICDataset`
- Load và transform images
- Return (image, label, image_id)

### `transforms.py`
- Training augmentation (random crop, flip, rotation, color jitter)
- Validation preprocessing (resize, center crop)
- ImageNet normalization

### `model.py`
- Build EfficientNet-B1 từ timm
- Count parameters
- Load/save checkpoints

### `train.py`
- Training loop cho 1 epoch
- WeightedRandomSampler cho imbalanced data
- Optimizer, scheduler, criterion setup
- DataLoader creation

### `evaluate.py`
- Evaluation trên val/test sets
- Confusion matrix plotting
- Classification report
- Create submission CSV
- Single image inference

## 📚 Usage Examples

### Load và sử dụng trained model

```python
from config import *
from model import build_model, load_checkpoint
from transforms import get_val_transform
from evaluate import predict_single_image

# Load model
model = build_model(num_classes=7)
model = model.to(DEVICE)
checkpoint = load_checkpoint(model, None, MODEL_PATH, DEVICE)

# Predict single image
image_path = "path/to/image.jpg"
transform = get_val_transform()
idx2label = checkpoint['idx2label']

image, top_labels, top_probs = predict_single_image(
    model, image_path, transform, DEVICE, idx2label, top_k=3
)

print("Top 3 predictions:")
for i, (label, prob) in enumerate(zip(top_labels, top_probs)):
    print(f"{i+1}. {label}: {prob:.4f}")
```

## ⚠️ Lưu ý

1. **Không push dữ liệu lên GitHub**: Folders `GroundTruth/` và `Input/` đã được thêm vào `.gitignore`

2. **Checkpoint cũ không tương thích**: Nếu có checkpoint từ EfficientNet-B3, cần train lại với B1

3. **Memory**: Nếu bị out of memory, giảm `BATCH_SIZE` trong `config.py`

4. **Windows**: `NUM_WORKERS = 0` để tránh lỗi multiprocessing

## 🎓 Dataset

ISIC 2018 Task 3: Lesion Diagnosis
- 7 classes: MEL, NV, BCC, AKIEC, BKL, DF, VASC
- Highly imbalanced (sử dụng WeightedRandomSampler)

## 📄 License

Dự án học tập - ISIC 2018 Challenge
