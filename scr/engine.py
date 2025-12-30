import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from .model import save_checkpoint
from .evaluate import evaluate

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, config_dict, start_epoch=0, best_val_f1=0.0, 
                best_epoch=0, history=None):

    if history is None:
        history = {
            'epoch': [], 'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_acc': [], 'val_f1': [], 
            'val_bal_acc': [], 'lr': []
        }
    
    end_epoch = start_epoch + num_epochs
    patience_counter = 0
    
    # Extract values from config or args
    val_every = config_dict.get('VAL_EVERY_N_EPOCHS', 1)
    save_every = config_dict.get('SAVE_EVERY_N_EPOCHS', 1)
    patience = config_dict.get('EARLY_STOP_PATIENCE', 5)
    use_cosine = config_dict.get('USE_COSINE_SCHEDULER', True)
    model_path = config_dict.get('MODEL_PATH', 'model.pt')
    num_classes = config_dict.get('NUM_CLASSES', 7)
    label2idx = config_dict.get('label2idx', {})
    idx2label = config_dict.get('idx2label', {})

    print(f"Training for {num_epochs} epochs (from {start_epoch} to {end_epoch})")

    for epoch in range(start_epoch, end_epoch):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{end_epoch}")
        print('='*60)
        
        # Train
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels, _ in train_pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)
        
        print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Validate only every N epochs
        should_validate = (epoch + 1) % val_every == 0 or (epoch + 1) == end_epoch
        
        if should_validate:
            print(" Running validation...")
            val_loss, val_acc, val_f1, val_bal_acc, _, _, _, _ = evaluate(model, val_loader, criterion, device)
            print(f" Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            # Check for best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                patience_counter = 0
                print(f"New best model! F1: {val_f1:.4f}")
            else:
                patience_counter += 1
                print(f" No improvement ({patience_counter}/{patience})")
            
            # Record history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['val_bal_acc'].append(val_bal_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Update learning rate
        if use_cosine:
            scheduler.step()
        
        # Save checkpoint periodically
        should_save = (epoch + 1) % save_every == 0 or (epoch + 1) == end_epoch
        if should_save:
            save_checkpoint(model, optimizer, epoch + 1, best_val_f1, best_epoch,
                           val_f1 if should_validate else 0.0, 
                           val_acc if should_validate else 0.0, 
                           history, label2idx, idx2label, num_classes, model_path)
        
        # Early stopping (only check when we validate)
        if should_validate and patience_counter >= patience:
            print(f"\n Early stopping at epoch {epoch + 1}")
            break

    print(f"\n{'='*60}")
    print(f" Training Stage Complete!")
    print(f" Best: Epoch {best_epoch} | F1: {best_val_f1:.4f}")
    print('='*60)
    
    return model, history, best_val_f1, best_epoch
