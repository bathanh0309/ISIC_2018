import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
import numpy as np
from typing import Tuple, Optional


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                   optimizer: torch.optim.Optimizer, device: torch.device,
                   scaler: Optional[GradScaler] = None) -> Tuple[float, float]:

    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def create_weighted_sampler(df, label_col: str = 'label_idx'):

    class_counts = df[label_col].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in df[label_col]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    print("Created WeightedRandomSampler for balanced training")
    return sampler


def get_optimizer(model: nn.Module, lr: float = 3e-4, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer for trainable parameters only.
    This supports transfer learning where backbone is frozen.
    """
    # Only optimize parameters that require gradients (trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    
    print(f"Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
    print(f"  Optimizing {trainable_count:,} / {total_params:,} parameters ({100*trainable_count/total_params:.1f}%)")
    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int, 
                 use_cosine: bool = True):

    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
        print(f"Scheduler: CosineAnnealingLR (T_max={num_epochs})")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        print("Scheduler: ReduceLROnPlateau (mode=max, factor=0.5, patience=2)")
    
    return scheduler


def get_criterion(use_label_smoothing: bool = False, label_smoothing: float = 0.1) -> nn.Module:

    if use_label_smoothing:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"Loss: CrossEntropyLoss with label smoothing ({label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Loss: CrossEntropyLoss")
    
    return criterion


def create_dataloaders(df_train, df_val, df_test, train_dataset, val_dataset, test_dataset,
                      batch_size: int, num_workers: int, use_weighted_sampler: bool):

    # Create WeightedRandomSampler if needed
    if use_weighted_sampler:
        sampler = create_weighted_sampler(df_train)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print("\n" + "=" * 60)
    print("DATA LOADERS")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("=" * 60)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from config import DEVICE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS
    from model import build_model
    
    # Test training utilities
    model = build_model(num_classes=7)
    model = model.to(DEVICE)
    
    optimizer = get_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, NUM_EPOCHS, use_cosine=True)
    criterion = get_criterion(use_label_smoothing=False)
    
    print("\n✓ Training utilities created successfully!")
