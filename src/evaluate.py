
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    classification_report, balanced_accuracy_score
)
from typing import Tuple, List, Dict


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
            device: torch.device, use_tta: bool = False) -> Tuple:

    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_image_ids = []
    
    with torch.no_grad():
        for images, labels, image_ids in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            if use_tta:
                # Basic TTA: Original + Horizontal Flip + Vertical Flip
                # 1. Original
                outputs = model(images)
                
                # 2. Horizontal Flip
                outputs_hf = model(torch.flip(images, dims=[3]))
                
                # 3. Vertical Flip
                outputs_vf = model(torch.flip(images, dims=[2]))
                
                # Average probabilities
                probs = (F.softmax(outputs, dim=1) + 
                         F.softmax(outputs_hf, dim=1) + 
                         F.softmax(outputs_vf, dim=1)) / 3.0
                
                # For loss calculation, use original outputs
                loss = criterion(outputs, labels)
                
                probs = probs.cpu().numpy()
                preds = np.argmax(probs, axis=1)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
            
            running_loss += loss.item() * images.size(0)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_image_ids.extend(image_ids)
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    epoch_bal_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, epoch_f1, epoch_bal_acc, all_preds, all_labels, all_probs, all_image_ids


def plot_confusion_matrix(labels: List[int], preds: List[int], idx2label: Dict,
                         save_path: str, title: str = 'Confusion Matrix'):

    num_classes = len(idx2label)
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[idx2label[i] for i in range(num_classes)],
                yticklabels=[idx2label[i] for i in range(num_classes)])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved: {save_path}")


def print_classification_report(labels: List[int], preds: List[int], 
                               idx2label: Dict) -> str:

    num_classes = len(idx2label)
    report = classification_report(
        labels, preds,
        target_names=[idx2label[i] for i in range(num_classes)],
        digits=4
    )
    print("\n=== CLASSIFICATION REPORT ===")
    print(report)
    return report


def create_submission(image_ids: List[str], preds: List[int], probs: List[np.ndarray],
                     idx2label: Dict, save_path: str) -> pd.DataFrame:

    num_classes = len(idx2label)
    
    submission_data = {
        'image_id': image_ids,
        'predicted_label': [idx2label[pred] for pred in preds],
    }
    
    # Add probability columns for each class
    for i in range(num_classes):
        submission_data[f'prob_{idx2label[i]}'] = [probs[j][i] for j in range(len(probs))]
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(save_path, index=False)
    
    print(f"\n✓ Submission file saved: {save_path}")
    print(f"  Shape: {submission_df.shape}")
    print(f"  First 5 predictions:")
    print(submission_df.head())
    
    return submission_df


def predict_single_image(model: nn.Module, image_path: str, transform,
                        device: torch.device, idx2label: Dict, 
                        top_k: int = 3, use_tta: bool = False) -> Tuple:

    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        if use_tta:
            # Original
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            
            # Horizontal Flip
            outputs_hf = model(torch.flip(image_tensor, dims=[3]))
            probs += F.softmax(outputs_hf, dim=1)
            
            # Vertical Flip
            outputs_vf = model(torch.flip(image_tensor, dims=[2]))
            probs += F.softmax(outputs_vf, dim=1)
            
            probs = (probs / 3.0).cpu().numpy()[0]
        else:
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
    
    # Get top-k predictions
    top_k_indices = np.argsort(probs)[::-1][:top_k]
    top_k_probs = probs[top_k_indices]
    top_k_labels = [idx2label[idx] for idx in top_k_indices]
    
    return image, top_k_labels, top_k_probs


def visualize_predictions(model: nn.Module, image_paths: List[Tuple[str, str]],
                         transform, device: torch.device, idx2label: Dict,
                         save_path: str, top_k: int = 3):

    n_images = len(image_paths)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    if n_images == 1:
        axes = [axes]
    
    print("\n=== INFERENCE DEMO ===\n")
    
    for idx, (img_path, img_name) in enumerate(image_paths):
        if os.path.exists(img_path):
            image, top_labels, top_probs = predict_single_image(
                model, img_path, transform, device, idx2label, top_k=top_k
            )
            
            # Display
            axes[idx].imshow(image)
            title = f"{img_name}\n"
            for i, (label, prob) in enumerate(zip(top_labels, top_probs)):
                title += f"{i+1}. {label}: {prob:.3f}\n"
            axes[idx].set_title(title, fontsize=9)
            axes[idx].axis('off')
            
            # Print to console
            print(f"Image: {img_name}")
            for i, (label, prob) in enumerate(zip(top_labels, top_probs)):
                print(f"  {i+1}. {label}: {prob:.4f} ({prob*100:.2f}%)")
            print()
        else:
            print(f"Image not found: {img_path}\n")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Inference demo saved: {save_path}")


if __name__ == "__main__":
    from .config import DEVICE, NUM_CLASSES
    from .model import build_model
    
    # Test evaluation utilities
    model = build_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    
    print("✓ Evaluation utilities loaded successfully!")
