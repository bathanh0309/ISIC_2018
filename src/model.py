"""
Model building utilities for ISIC 2018 Classification.
Supports Transfer Learning with frozen backbone.
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Tuple, Optional


def build_model(num_classes: int = 7, pretrained: bool = True, model_name: str = 'efficientnet_b1',
                drop_rate: float = 0.3, drop_path_rate: float = 0.2,
                freeze_backbone: bool = False) -> nn.Module:

    # Create model with pretrained weights
    model = timm.create_model(
        model_name, 
        pretrained=pretrained, 
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )
    
    # Freeze backbone if requested (transfer learning)
    if freeze_backbone:
        freeze_backbone_layers(model)
        print(f"✓ Backbone frozen - only classifier will be trained")
    
    return model


def freeze_backbone_layers(model: nn.Module) -> None:
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier layer
    # For timm EfficientNet, classifier is usually model.classifier or model.fc
    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        # Fallback: unfreeze last layer
        params = list(model.parameters())
        if params:
            params[-1].requires_grad = True
            params[-2].requires_grad = True  # bias and weight

def unfreeze_backbone_layers(model: nn.Module, num_layers: int = -1) -> None:
    if num_layers == -1:
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
        print("✓ All layers unfrozen for fine-tuning")
    else:
        # Unfreeze last N layers
        params = list(model.parameters())
        for param in params[-num_layers:]:
            param.requires_grad = True
        print(f"✓ Last {num_layers} layers unfrozen for fine-tuning")


def get_trainable_params(model: nn.Module) -> Tuple[int, int]:

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Alias for get_trainable_params for backward compatibility."""
    return get_trainable_params(model)


def load_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], 
                    checkpoint_path: str, device: torch.device) -> Dict:

    print(f"Loading checkpoint from: {checkpoint_path}")
    # weights_only=False is needed for PyTorch 2.6+ because checkpoint contains
    # numpy arrays in history dict. This is safe since we created the checkpoint ourselves.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_val_f1 = checkpoint.get('best_val_f1', 0.0)
    
    print(f"✓ Loaded checkpoint from epoch {epoch}, best F1: {best_val_f1:.4f}")
    
    return checkpoint


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, best_val_f1: float, best_epoch: int,
                   val_f1: float, val_acc: float, history: Dict,
                   label2idx: Dict, idx2label: Dict, num_classes: int,
                   checkpoint_path: str) -> None:
    """
    Save model checkpoint with training state.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch,
        'val_f1': val_f1,
        'val_acc': val_acc,
        'history': history,
        'label2idx': label2idx,
        'idx2label': idx2label,
        'num_classes': num_classes,
    }, checkpoint_path)
    
    print(f"✓ Checkpoint saved: {checkpoint_path}")


def print_model_info(model: nn.Module, model_name: str = 'EfficientNet-B1'):
    """Print model information including trainable parameters."""
    total_params, trainable_params = count_parameters(model)
    frozen_params = total_params - trainable_params
    
    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Training mode: {'Transfer Learning (frozen backbone)' if frozen_params > 0 else 'Full Training'}")
    print("=" * 60)


if __name__ == "__main__":
    from config import NUM_CLASSES, DEVICE
    
    print("=" * 60)
    print("Testing Transfer Learning Model Building")
    print("=" * 60)
    
    # Test 1: Full training (no freeze)
    print("\n--- Test 1: Full Training ---")
    model_full = build_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False)
    model_full = model_full.to(DEVICE)
    print_model_info(model_full, 'EfficientNet-B1 (Full Training)')
    
    # Test 2: Transfer learning (frozen backbone)
    print("\n--- Test 2: Transfer Learning (Frozen Backbone) ---")
    model_frozen = build_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
    model_frozen = model_frozen.to(DEVICE)
    print_model_info(model_frozen, 'EfficientNet-B1 (Transfer Learning)')
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    output = model_frozen(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    print("\n✓ Model building tests passed!")
