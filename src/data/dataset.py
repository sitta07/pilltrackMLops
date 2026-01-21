# src/data/dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(img_size, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

def create_dataloaders(train_dir, val_dir, config):
    # 1. Setup Transforms
    train_transform = get_transforms(config['img_size'], is_train=True)
    val_transform = get_transforms(config['img_size'], is_train=False)

    # 2. Load Datasets
    # ImageFolder จะอ่านชื่อ Folder เป็น Class อัตโนมัติ
    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    # 3. Create Loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers']
    )

    return train_loader, val_loader, train_ds.classes