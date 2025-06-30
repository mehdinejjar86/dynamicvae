import os
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import numpy as np
import torchvision.transforms as T

class UniversalImageDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_mode='grayscale',  # 'grayscale' or 'color'
        target_size=(64, 64),
        transform=None,
        log_to_wandb=False
    ):
        """
        Args:
            root_dir (string): Directory with all images
            image_mode (str): 'grayscale' or 'color'
            target_size (tuple): (height, width) for resizing
            transform (callable, optional): Optional transforms
            log_to_wandb (bool): Enable WandB logging
        """
        self.root_dir = Path(root_dir)
        self.image_mode = image_mode
        self.log_to_wandb = log_to_wandb
        self.target_size = target_size
        
        # Collect all image paths
        self.image_paths = []
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        for ext in extensions:
            self.image_paths.extend(list(self.root_dir.glob(ext)))
            
        # Default transforms if none provided
        self.transform = transform or T.Compose([
            T.Resize(self.target_size),
            T.ToTensor(),
        ])
        
        # Set up mode conversion
        self.mode_conversion = {
            'grayscale': {'load': 'RGB', 'convert': 'L'},
            'color': {'load': 'RGB', 'convert': 'RGB'}
        }
        
        if self.log_to_wandb:
            wandb.init(project="VAE-Dataset", config={
                "dataset_path": str(root_dir),
                "image_mode": image_mode,
                "target_size": target_size,
                "num_images": len(self.image_paths)
            })
            self._log_dataset_stats()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image with unified channel management
        img = Image.open(img_path).convert(
            self.mode_conversion[self.image_mode]['load']
        )
        
        # Convert to desired mode
        if self.image_mode == 'grayscale':
            img = img.convert(self.mode_conversion[self.image_mode]['convert'])
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
            
        # Add channel dimension if missing (for grayscale)
        if img.ndim == 2:
            img = img.unsqueeze(0)
            
        if self.log_to_wandb and idx == 0:
            wandb.log({"sample_image": wandb.Image(img.numpy().transpose(1,2,0))})
            
        return img

    def _log_dataset_stats(self):
        """Log dataset statistics to WandB"""
        # Log first 10 images
        sample_images = [self[i] for i in range(min(10, len(self)))]
        wandb.log({
            "dataset_samples": [
                wandb.Image(img.numpy().transpose(1,2,0), 
                          caption=f"Image {i}")
                for i, img in enumerate(sample_images)
            ]
        })
        
        # Log size distribution
        sizes = []
        for path in self.image_paths:
            with Image.open(path) as img:
                sizes.append(img.size)
        wandb.log({
            "width_distribution": wandb.Histogram([w for w, h in sizes]),
            "height_distribution": wandb.Histogram([h for w, h in sizes])
        })

class GlandImageDataset(Dataset):
    def __init__(
        self,
        root_dir,
        target_size=(64, 64),
        min_black_ratio=0.5,
        image_mode='grayscale',  # 'grayscale' or 'color'
        log_to_wandb=False,
        max_images_to_log=10
    ):
        """
        Args:
            root_dir (str): Folder with grayscale images
            target_size (tuple): Final size of images (H, W)
            min_black_ratio (float): Minimum black pixel ratio to exclude image
            log_to_wandb (bool): Whether to log filtered images
        """
        assert image_mode == 'grayscale', "Only grayscale images are supported"
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.min_black_ratio = min_black_ratio
        self.log_to_wandb = log_to_wandb
        self.max_images_to_log = max_images_to_log

        self.transform = T.Compose([
            T.Resize(self.target_size),
            T.ToTensor(),  # Converts to float32 and scales to [0, 1]
        ])
        
        # Load and filter image paths
        self.image_paths = self._filter_images()

        if self.log_to_wandb:
            wandb.init(project="Gland-Dataset", config={
                "dataset_path": str(root_dir),
                "target_size": target_size,
                "kept_images": len(self.image_paths)
            })
            self._log_sample_images()

    def _filter_images(self):
        """Filter out images with too many black pixels"""
        valid_paths = []
        black_threshold = 5  # pixel intensity <= this is considered "black"

        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            for path in self.root_dir.glob(ext):
                img = Image.open(path).convert('L')  # grayscale
                img_resized = img.resize(self.target_size)
                img_np = np.array(img_resized, dtype=np.uint8)

                num_black_pixels = np.sum(img_np <= black_threshold)
                total_pixels = img_np.size
                black_pixel_ratio = num_black_pixels / total_pixels

                if black_pixel_ratio < self.min_black_ratio:
                    valid_paths.append(path)

        return valid_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')
        img = self.transform(img)  # shape (1, H, W), range [0,1]

        # Min-max scale per image (clip to avoid division by 0)
        min_val = img.min()
        max_val = img.max()
        img = (img - min_val) / (max_val - min_val + 1e-8)

        return img

    def _log_sample_images(self):
        """Log a few samples to WandB"""
        samples = [self[i] for i in range(min(self.max_images_to_log, len(self)))]
        wandb.log({
            "filtered_samples": [
                wandb.Image(img.squeeze(0).numpy(), caption=f"Image {i}")
                for i, img in enumerate(samples)
            ]
        })

# Example usage
if __name__ == "__main__":
    # Initialize WandB (optional)
    wandb.init(project="VAE-Data-Test")

    # Create dataset
    dataset = UniversalImageDataset(
        root_dir="/home/nightstalker/Projects/ml/vae/train",
        image_mode='grayscale',  # Try 'color' for RGB
        target_size=(28, 28),
        log_to_wandb=True
    )
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Test one batch
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}")
    
