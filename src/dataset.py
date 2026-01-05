
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Tuple


class ISICDataset(Dataset):

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:

        row = self.df.iloc[idx]
        image_path = row['image_path']
        label = row['label_idx']
        image_id = row['image_id']
        
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long), image_id


if __name__ == "__main__":
    # Test dataset creation
    import pandas as pd
    from transforms import get_train_transform, get_val_transform
    from config import DIR_TRAIN_IMG
    
    # Create dummy dataframe
    dummy_df = pd.DataFrame({
        'image_id': ['ISIC_0024306'],
        'label': ['NV'],
        'image_path': [os.path.join(DIR_TRAIN_IMG, 'ISIC_0024306.jpg')],
        'label_idx': [0]
    })
    
    # Create dataset
    dataset = ISICDataset(dummy_df, transform=get_train_transform())
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test loading
    if len(dataset) > 0 and os.path.exists(dummy_df.iloc[0]['image_path']):
        img, label, img_id = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Label: {label}")
        print(f"Image ID: {img_id}")
