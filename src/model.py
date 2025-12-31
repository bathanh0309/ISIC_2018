import torch
import torch.nn as nn
import timm
from typing import Dict, Tuple, Optional


def build_model(num_classes: int = 7, pretrained: bool = True, model_name: str = 'efficientnet_b1',
                drop_rate: float = 0.0, drop_path_rate: float = 0.0) -> nn.Module:

    model = timm.create_model(
        model_name, 
        pretrained=pretrained, 
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )
    return model

def count_parameters(model: nn.Module) -> Tuple[int, int]:

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def load_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], 
                    checkpoint_path: str, device: torch.device) -> Dict:

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_val_f1 = checkpoint.get('best_val_f1', 0.0)
    
    print(f"Loaded checkpoint from epoch {epoch}, best F1: {best_val_f1:.4f}")
    
    return checkpoint


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, best_val_f1: float, best_epoch: int,
                   val_f1: float, val_acc: float, history: Dict,
                   label2idx: Dict, idx2label: Dict, num_classes: int,
                   checkpoint_path: str) -> None:

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
    
    print(f"Checkpoint saved: {checkpoint_path}")


def print_model_info(model: nn.Module, model_name: str = 'EfficientNet-B1'):
    """Print model information."""
    total_params, trainable_params = count_parameters(model)
    
    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)


if __name__ == "__main__":
    # Test model creation
    from .config import NUM_CLASSES, DEVICE
    
    model = build_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)
    
    print_model_info(model, 'EfficientNet-B1')
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 240, 240).to(DEVICE)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model created successfully!")
