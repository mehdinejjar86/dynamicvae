import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
from tqdm import tqdm
import torchvision
import torch.nn as nn
from datetime import datetime
from vae import DynamicVAE
from dataset import UniversalImageDataset
import argparse

class VAETrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.checkpoint_dir = os.path.join(config['save_dir'], self.time)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model = DynamicVAE(
            input_height=config['input_height'],
            input_width=config['input_width'],
            encoder_channels=config['encoder_channels'],
            decoder_channels=config['decoder_channels'],
            latent_dim=config['latent_dim']
        ).to(self.device)

        if self.device.type == "cuda" and torch.cuda.device_count() > 1 and config.get('use_dataparallel', True):
            print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

        full_dataset = UniversalImageDataset(
            root_dir=config['data_path'],
            image_mode=config['image_mode'],
            target_size=(config['input_height'], config['input_width'])
        )
        print(f"Loaded {len(full_dataset)} images")
        train_size = int(0.9 * len(full_dataset))
        print(f"Train size: {train_size}")
        val_size = len(full_dataset) - train_size
        print(f"Val size: {val_size}")
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4)
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4)
        )

        wandb.init(
            project=config['project_name'],
            config=config,
            tags=config['tags']
        )
        wandb.watch(self.model, log='all', log_freq=100)

        self.best_loss = float('inf')
        self.current_epoch = 0
        self.epoch_metrics = []

    def train_epoch(self):
        self.model.train()
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        total_loss = 0.0
        total_recon_loss = 0.0

        for batch_idx, images in enumerate(progress_bar):
            images = images.to(self.device)

            recon_images, mu, logvar = self.model(images)
            loss = self.model.module.loss_function(recon_images, images, mu, logvar) if isinstance(self.model, nn.DataParallel) else self.model.loss_function(recon_images, images, mu, logvar)
            recon_loss = nn.functional.mse_loss(recon_images, images, reduction='sum')

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()

            progress_bar.set_postfix({
                'loss': loss.item() / len(images),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            if self.current_epoch % self.config['image_log_interval'] == 0:
                self.log_reconstructions(images, recon_images, "Training Reconstructions")

            if batch_idx % self.config['log_interval'] == 0:
                wandb.log({
                    "train/batch_loss": loss.item() / len(images),
                    "train/recon_loss": recon_loss.item() / len(images),
                    "train/lr": self.optimizer.param_groups[0]['lr']
                })

        avg_loss = total_loss / len(self.train_dataset)
        avg_recon = total_recon_loss / len(self.train_dataset)
        return avg_loss, avg_recon

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        with torch.no_grad():
            for images in self.val_loader:
                images = images.to(self.device)
                recon_images, mu, logvar = self.model(images)
                loss = self.model.module.loss_function(recon_images, images, mu, logvar) if isinstance(self.model, nn.DataParallel) else self.model.loss_function(recon_images, images, mu, logvar)
                recon_loss = nn.functional.mse_loss(recon_images, images, reduction='sum')
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()

                if self.current_epoch % self.config['image_log_interval'] == 0:
                    self.log_reconstructions(images, recon_images, "Validation Reconstructions")

        avg_loss = total_loss / len(self.val_dataset)
        avg_recon = total_recon_loss / len(self.val_dataset)
        wandb.log({
            "val/loss": avg_loss,
            "val/recon_loss": avg_recon
        })
        self.scheduler.step(avg_loss)
        return avg_loss, avg_recon

    def log_reconstructions(self, original, reconstructed, title="Reconstructions"):
        original = original[:8].cpu()
        reconstructed = reconstructed[:8].cpu()
        comparison = torch.cat([original, reconstructed], dim=0)
        grid = torchvision.utils.make_grid(comparison, nrow=8)
        wandb.log({
            title: wandb.Image(grid.numpy().transpose(1, 2, 0)),
            "epoch": self.current_epoch
        })

    def save_metrics(self):
        metrics_path = os.path.join(self.checkpoint_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.epoch_metrics, f, indent=2)

    def save_checkpoint(self, is_best):
        state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        state = {
            'epoch': self.current_epoch,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }
        torch.save(state, os.path.join(self.checkpoint_dir, "last_checkpoint.pth"))
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, "best_model.pth"))
            wandb.save(os.path.join(self.checkpoint_dir, "best_model.pth"))

    def train(self):
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            train_loss, train_recon = self.train_epoch()
            val_loss, val_recon = self.validate()

            wandb.log({
                "train/epoch_loss": train_loss,
                "train/recon_loss": train_recon,
                "val/loss": val_loss,
                "val/recon_loss": val_recon,
                "epoch": epoch
            })

            self.epoch_metrics.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_recon_loss": train_recon,
                "val_loss": val_loss,
                "val_recon_loss": val_recon,
                "lr": self.optimizer.param_groups[0]['lr']
            })
            self.save_metrics()

            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            self.save_checkpoint(is_best)

            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Dynamic VAE model.")
    parser.add_argument('--config', type=str, default='configs/vae_config.json',
                        help='Path to the JSON configuration file.')
    args = parser.parse_args()

    # Load configuration from the specified JSON file
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    os.makedirs(config['save_dir'], exist_ok=True)
    trainer = VAETrainer(config)
    trainer.train()
