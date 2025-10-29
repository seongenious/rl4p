#!/usr/bin/env python3
"""
Guardian training script for RL4P project
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
import pickle

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import argparse
from tqdm import tqdm
from pathlib import Path

from model.guardian import Guardian
from configs import EnvConfig, ModelConfig


class CollisionDataset(Dataset):
    """Dataset for collision prediction"""
    
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Extract features and labels
        bev_features = torch.FloatTensor(sample['bev_features'])
        action = torch.FloatTensor(sample['action'])  # [delta, speed]
        collision = torch.FloatTensor([float(sample['collision'])])
        
        return {
            'bev_features': bev_features,
            'action': action,
            'collision': collision
        }


def load_data(data_path="../data/collision/data.pkl"):
    """Load collision data"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in tqdm(train_loader, desc="training ..."):
        # Get batch data
        bev_features = batch['bev_features'].to(device)
        action = batch['action'].to(device)
        collision = batch['collision'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_collision = model(bev_features, action)
        loss = criterion(pred_collision, collision)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred_binary = (pred_collision > 0.5).float()
        total_correct += (pred_binary == collision).sum().item()
        total_samples += collision.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def calculate_metrics(predictions, targets):
    """Calculate precision, recall, f1, and auc without sklearn"""
    predictions = np.array(predictions)
    targets = np.array(targets)
    pred_binary = (predictions > 0.5).astype(int)
    
    # Calculate precision
    tp = np.sum((pred_binary == 1) & (targets == 1))
    fp = np.sum((pred_binary == 1) & (targets == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Calculate recall
    fn = np.sum((pred_binary == 0) & (targets == 1))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Calculate F1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate AUC (simplified version)
    # Sort by prediction scores
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_targets = targets[sorted_indices]
    
    # Calculate AUC using trapezoidal rule
    auc = 0.0
    tp_count = 0
    fp_count = 0
    
    for i, target in enumerate(sorted_targets):
        if target == 1:
            tp_count += 1
        else:
            fp_count += 1
            auc += tp_count
    
    total_positives = np.sum(targets)
    total_negatives = len(targets) - total_positives
    
    if total_positives > 0 and total_negatives > 0:
        auc = auc / (total_positives * total_negatives)
    else:
        auc = 0.5  # Random performance
    
    return precision, recall, f1, auc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="validating ..."):
            # Get batch data
            bev_features = batch['bev_features'].to(device)
            action = batch['action'].to(device)
            collision = batch['collision'].to(device)
            
            # Forward pass
            pred_collision = model(bev_features, action)
            loss = criterion(pred_collision, collision)
            
            # Statistics
            total_loss += loss.item()
            pred_binary = (pred_collision > 0.5).float()
            total_correct += (pred_binary == collision).sum().item()
            total_samples += collision.size(0)
            
            # Store predictions for metrics
            predictions.extend(pred_collision.cpu().numpy())
            targets.extend(collision.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples
    
    # Calculate additional metrics
    precision, recall, f1, auc = calculate_metrics(predictions, targets)
    
    return avg_loss, accuracy, precision, recall, f1, auc


def save_model(model, epoch, loss, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }
    
    save_path = os.path.join(save_dir, f'guardian_epoch_{epoch}.pt')
    torch.save(checkpoint, save_path)
    
    # Also save the complete model
    model_path = os.path.join(save_dir, 'guardian.pt')
    torch.save(model, model_path)


def plot(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Guardian for RL4P')
    parser.add_argument('--data_path', type=str, default="../data/collision/data.pkl",
                       help='Path to collision data')
    parser.add_argument('--save_dir', type=str, default="../ckpt/guardian",
                       help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--test_size', type=float, default=0.1,
                       help='Test set size ratio')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Validation set size ratio')
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")

    # Load data
    data = load_data(args.data_path)
    
    # Split data manually without sklearn
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    
    test_size = int(len(data) * args.test_size)
    val_size = int(len(data) * args.val_size)
    
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create datasets and dataloaders
    train_dataset = CollisionDataset(train_data)
    val_dataset = CollisionDataset(val_data)
    test_dataset = CollisionDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    print("Creating Guardian model...")
    config = ModelConfig()
    model = Guardian(
      bev_dim=config.guardian.bev_feat_dim, 
      action_dim=config.guardian.action_feat_dim, 
      hidden_dim=config.guardian.hidden_dim).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    
    # Training loop
    print("Start training...")
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validate
        val_loss, val_acc, precision, recall, f1, auc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.3f}")
        print(f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.3f}")
        print(f"Val Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, epoch+1, val_loss, args.save_dir)
            print(f"New best model saved! (Val Loss: {val_loss:.6f})")
        
        # Plot training curves every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_dir = Path(args.save_dir) / 'train'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / 'training_curves.png'
            plot(train_losses, val_losses, train_accuracies, val_accuracies, save_path)
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = validate_epoch(model, test_loader, criterion, device)
    
    print(f"\n=== Final Test Results ===")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    print(f"Test Precision: {test_precision:.3f}")
    print(f"Test Recall: {test_recall:.3f}")
    print(f"Test F1-Score: {test_f1:.3f}")
    print(f"Test AUC: {test_auc:.3f}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
