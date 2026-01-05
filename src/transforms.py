
import torchvision.transforms as T
from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transform():

    transform = T.Compose([
        T.Resize(int(IMG_SIZE * 1.15)),  # Resize slightly larger
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(20),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05  # Reduced from 0.1 to avoid extreme color shifts
        ),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
    ])
    
    return transform


def get_val_transform():

    transform = T.Compose([
        T.Resize(int(IMG_SIZE * 1.15)),  # Resize slightly larger
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return transform


def print_transforms():
    """Print the transform pipelines."""
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    print("=" * 60)
    print("DATA TRANSFORMS")
    print("=" * 60)
    print(f"\nTrain transforms:\n{train_transform}")
    print(f"\nVal/Test transforms:\n{val_transform}")
    print("=" * 60)


if __name__ == "__main__":
    print_transforms()
