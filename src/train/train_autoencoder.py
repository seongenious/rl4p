#!/usr/bin/env python3
"""
AutoEncoder training script for RL4P project
"""

import os
import sys
sys.path.append('..')
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from PIL import Image
import glob
import argparse
from tqdm import tqdm
from pathlib import Path

from model.autoencoder import AutoEncoder
from configs import EnvConfig, ActorConfig


class ImageDataset(Dataset):
    """Custom dataset for loading images from rl4p/data/img"""
    
    def __init__(self, data_dir, img_size=(64, 64), transform=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        
        # Get all image files
        self.image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
        print(f"Found {len(self.image_paths)} images in {data_dir}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Resize image
        # image = image.resize(self.img_size)
        
        # Convert to tensor and normalize
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        if self.transform:
            image = self.transform(image)
        
        return image


def create_data_loaders(data_dir, batch_size=32, img_size=(64, 64), train_split=0.8):
    """Create train and validation data loaders"""
    
    # Create full dataset
    full_dataset = ImageDataset(data_dir, img_size)
    
    # Split dataset
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, images in enumerate(tqdm(train_loader, desc="training ...")):
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed = model(images)
        loss = criterion(images, reconstructed)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images in tqdm(val_loader, desc="validating ..."):
            images = images.to(device)
            
            # Forward pass
            reconstructed = model(images)
            loss = criterion(images, reconstructed)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def save_model(model, epoch, loss, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }
    
    save_path = os.path.join(save_dir, f'autoencoder_epoch_{epoch}.pt')
    torch.save(checkpoint, save_path)
    
    # Also save the complete model
    model_path = os.path.join(save_dir, 'autoencoder.pt')
    torch.save(model, model_path)


def visualize(model, val_loader, device, save_path, num_samples=5):
    """Visualize original and reconstructed images"""
    model.eval()
    
    with torch.no_grad():
        # Get a batch of validation images
        images = next(iter(val_loader))[:num_samples].to(device)
        reconstructed = model(images)
        
        # Move to CPU and convert to numpy
        images = images.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        
        for i in range(num_samples):
            # Original image
            orig_img = images[i].transpose(1, 2, 0)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstructed image
            recon_img = reconstructed[i].transpose(1, 2, 0)
            axes[1, i].imshow(recon_img)
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot(train_losses, val_losses, save_path):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train AutoEncoder for RL4P')
    parser.add_argument('--data_dir', type=str, default='../data/img', 
                       help='Path to image data directory')
    parser.add_argument('--save_dir', type=str, default='../ckpt/autoencoder',
                       help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")

    # Create config
    config = EnvConfig()
    img_width = config.observation.img_size[0] // config.observation.img_downsample_rate
    img_height = config.observation.img_size[1] // config.observation.img_downsample_rate
    img_size = (img_width, img_height)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = create_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        img_size=(img_width, img_height)
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating AutoEncoder model...")
    model = AutoEncoder(
        img_dim=config.actor.img_shape,
        kernel_size=config.actor.kernel_size,
        embed_dim=config.actor.embed_dim,
        conv_dims=config.actor.conv_dims.copy(),
        fc_dims=config.actor.fc_dims.copy()
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Training loop
    print("Start training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, epoch+1, val_loss, args.save_dir)
            print(f"New best model saved! (Val Loss: {val_loss:.6f})")
        
        # Visualize reconstruction every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_dir = Path(args.save_dir) / 'eval'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f'epoch_{epoch+1}.png'
            visualize(model, val_loader, device, save_path)
        
        # Plot training curves every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_dir = Path(args.save_dir) / 'train'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / 'training_curves.png'
            plot(train_losses, val_losses, save_path)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()