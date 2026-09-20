import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

class AddGaussianNoise(object):
    """Simulates sensor noise on float tensors in range [0.0, 1.0]."""
    def __init__(self, std=0.03, p=0.3):
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(tensor) * self.std
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

def get_dataloaders(data_dir, batch_size=32, img_height=224, img_width=224, num_workers=2):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': v2.Compose([
            v2.Resize((img_height, img_width)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=45),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True), 
            AddGaussianNoise(std=0.03, p=0.3),
            v2.Normalize(mean=norm_mean, std=norm_std)
        ]),
        'val': v2.Compose([
            v2.Resize((img_height, img_width)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=norm_mean, std=norm_std)
        ]),
        'test': v2.Compose([
            v2.Resize((img_height, img_width)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=norm_mean, std=norm_std)
        ])
    }

    splits = ['train', 'val', 'test']
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x])
        for x in splits
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == 'train'),
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        for x in splits
    }
    
    class_names = image_datasets['train'].classes
    return dataloaders, class_names, image_datasets

