import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from PIL import Image

class ColorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.classes = sorted([d for d in os.listdir(root_dir)  
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # load all
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        print(f"Found {len(self.samples)} images in {len(self.classes)} classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (128, 128), (0, 0, 0)) 
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_color_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_data_loaders(data_dir, batch_size=32, val_split=0.15, test_split=0.15, num_workers=None):
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    train_transform, val_transform = get_color_transforms()
    
    full_dataset_for_split = ColorDataset(data_dir, transform=val_transform)
    
    # split
    total_size = len(full_dataset_for_split)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    if train_size <= 0 or val_size <=0 or test_size <=0:
        raise ValueError("Not enough samples for the specified split ratios, or one split resulted in 0 samples.")
    
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices, test_indices = random_split(
        range(total_size), [train_size, val_size, test_size], generator=generator
    )

    train_full_dataset_with_train_transform = ColorDataset(data_dir, transform=train_transform)
    train_dataset = Subset(train_full_dataset_with_train_transform, train_indices)
    
    val_dataset = Subset(full_dataset_for_split, val_indices)
    test_dataset = Subset(full_dataset_for_split, test_indices)
        
    train_loader = DataLoader(
        train_dataset,  
        batch_size=batch_size,  
        shuffle=True,  
        num_workers=num_workers,  
        pin_memory=torch.cuda.is_available(),
        drop_last=True
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
    
    print(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, full_dataset_for_split.classes