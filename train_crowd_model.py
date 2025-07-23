#!/usr/bin/env python3
"""
JHU Crowd v2.0 Training Pipeline
Comprehensive crowd counting model training with GPU acceleration
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import vgg16, resnet50
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform
import argparse
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import random
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JHUCrowdDataset(Dataset):
    """JHU Crowd v2.0 Dataset Loader"""
    
    def __init__(self, data_root, split='train', transform=None, target_transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load image paths and annotations
        self.images = []
        self.annotations = []
        
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        # Load images and ground truth
        images_dir = os.path.join(split_dir, 'images')
        gt_dir = os.path.join(split_dir, 'gt')
        
        if os.path.exists(images_dir):
            for img_file in os.listdir(images_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(images_dir, img_file)
                    
                    # Find corresponding annotation file
                    base_name = os.path.splitext(img_file)[0]
                    gt_file = os.path.join(gt_dir, f"{base_name}.txt")
                    
                    if os.path.exists(gt_file):
                        self.images.append(img_path)
                        self.annotations.append(gt_file)
        
        logger.info(f"Loaded {len(self.images)} images for {split} split")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations (head positions)
        ann_path = self.annotations[idx]
        points = self.load_annotations(ann_path)
        
        # Generate density map
        density_map = self.generate_density_map(image.shape[:2], points)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            density_map = self.target_transform(density_map)
        
        return image, density_map, len(points)
    
    def load_annotations(self, ann_path):
        """Load head positions from annotation file"""
        points = []
        try:
            with open(ann_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            x, y = float(parts[0]), float(parts[1])
                            points.append([x, y])
        except Exception as e:
            logger.warning(f"Error loading annotations from {ann_path}: {e}")
        
        return np.array(points) if points else np.empty((0, 2))
    
    def generate_density_map(self, image_shape, points, sigma=15):
        """Generate Gaussian density map from point annotations"""
        h, w = image_shape
        density_map = np.zeros((h, w), dtype=np.float32)
        
        if len(points) == 0:
            return density_map
        
        # Adaptive sigma based on local density
        if len(points) > 1:
            # Calculate distances to nearest neighbors
            distances = pdist(points)
            if len(distances) > 0:
                avg_distance = np.mean(distances)
                sigma = max(sigma, avg_distance / 3)
        
        # Generate Gaussian blobs for each point
        for point in points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < w and 0 <= y < h:
                # Create Gaussian kernel
                kernel_size = int(6 * sigma)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                # Generate 2D Gaussian
                kernel = np.zeros((kernel_size, kernel_size))
                center = kernel_size // 2
                
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        dist_sq = (i - center) ** 2 + (j - center) ** 2
                        kernel[i, j] = np.exp(-dist_sq / (2 * sigma ** 2))
                
                # Place kernel on density map
                y_start = max(0, y - center)
                y_end = min(h, y + center + 1)
                x_start = max(0, x - center)
                x_end = min(w, x + center + 1)
                
                kernel_y_start = max(0, center - y)
                kernel_y_end = kernel_y_start + (y_end - y_start)
                kernel_x_start = max(0, center - x)
                kernel_x_end = kernel_x_start + (x_end - x_start)
                
                density_map[y_start:y_end, x_start:x_end] += \
                    kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]
        
        return density_map

class CSRNet(nn.Module):
    """CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes"""
    
    def __init__(self, pretrained=True):
        super(CSRNet, self).__init__()
        
        # Frontend: VGG-16 backbone
        vgg = vgg16(pretrained=pretrained)
        features = list(vgg.features.children())
        
        # Remove max pooling layers to maintain spatial resolution
        self.frontend = nn.ModuleList(features[:23])
        
        # Backend: Dilated convolutions
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        # Frontend
        for layer in self.frontend:
            x = layer(x)
        
        # Backend
        x = self.backend(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class CANNet(nn.Module):
    """Context-Aware Network for Crowd Counting"""
    
    def __init__(self, pretrained=True):
        super(CANNet, self).__init__()
        
        # Frontend: ResNet-50 backbone
        resnet = resnet50(pretrained=pretrained)
        self.frontend = nn.Sequential(*list(resnet.children())[:-2])
        
        # Multi-scale context module
        self.context_module = nn.ModuleList([
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(2048, 512, kernel_size=3, padding=2, dilation=2),
            nn.Conv2d(2048, 512, kernel_size=3, padding=3, dilation=3),
            nn.Conv2d(2048, 512, kernel_size=3, padding=4, dilation=4)
        ])
        
        # Fusion and output
        self.fusion = nn.Conv2d(2048, 512, kernel_size=1)
        self.output = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        # Extract features
        features = self.frontend(x)
        
        # Multi-scale context
        context_features = []
        for conv in self.context_module:
            context_features.append(F.relu(conv(features)))
        
        # Concatenate and fuse
        fused = torch.cat(context_features, dim=1)
        fused = F.relu(self.fusion(fused))
        
        # Generate density map
        output = self.output(fused)
        
        # Upsample to original resolution
        output = F.interpolate(output, scale_factor=8, mode='bilinear', align_corners=False)
        
        return output
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class CrowdCountLoss(nn.Module):
    """Combined loss for crowd counting"""
    
    def __init__(self, mse_weight=1.0, ssim_weight=0.1):
        super(CrowdCountLoss, self).__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        # MSE loss on density maps
        mse_loss = self.mse_loss(pred, target)
        
        # Count loss (total sum)
        pred_count = torch.sum(pred.view(pred.size(0), -1), dim=1)
        target_count = torch.sum(target.view(target.size(0), -1), dim=1)
        count_loss = self.mse_loss(pred_count, target_count)
        
        return self.mse_weight * mse_loss + count_loss

def get_transforms(split='train'):
    """Get data transforms for training/validation"""
    if split == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_target_transforms():
    """Target transforms for density maps"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.interpolate(x.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0))
    ])

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} Training')
    
    for batch_idx, (images, density_maps, counts) in enumerate(pbar):
        images = images.to(device)
        density_maps = density_maps.to(device)
        counts = counts.to(device).float()
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, density_maps)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        pred_counts = torch.sum(outputs.view(outputs.size(0), -1), dim=1)
        mae = torch.mean(torch.abs(pred_counts - counts))
        
        running_loss += loss.item()
        running_mae += mae.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'MAE': f'{mae.item():.2f}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_mae = running_mae / len(dataloader)
    
    return epoch_loss, epoch_mae

def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} Validation')
        
        for images, density_maps, counts in pbar:
            images = images.to(device)
            density_maps = density_maps.to(device)
            counts = counts.to(device).float()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, density_maps)
            
            # Calculate metrics
            pred_counts = torch.sum(outputs.view(outputs.size(0), -1), dim=1)
            mae = torch.mean(torch.abs(pred_counts - counts))
            
            running_loss += loss.item()
            running_mae += mae.item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MAE': f'{mae.item():.2f}'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_mae = running_mae / len(dataloader)
    
    return epoch_loss, epoch_mae

def save_checkpoint(model, optimizer, epoch, loss, mae, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'mae': mae,
    }, filepath)

def main():
    parser = argparse.ArgumentParser(description='JHU Crowd v2.0 Training')
    parser.add_argument('--data_root', type=str, required=True,
                      help='Path to JHU Crowd v2.0 dataset')
    parser.add_argument('--model', type=str, default='csrnet', choices=['csrnet', 'cannet'],
                      help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU device to use')
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f'Using GPU: {device}')
    else:
        device = torch.device('cpu')
        logger.info('Using CPU')
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create datasets
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    target_transform = get_target_transforms()
    
    train_dataset = JHUCrowdDataset(
        args.data_root, 'train', 
        transform=train_transform, 
        target_transform=target_transform
    )
    
    val_dataset = JHUCrowdDataset(
        args.data_root, 'val',
        transform=val_transform,
        target_transform=target_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    if args.model == 'csrnet':
        model = CSRNet(pretrained=True)
    elif args.model == 'cannet':
        model = CANNet(pretrained=True)
    
    model = model.to(device)
    
    # Create loss and optimizer
    criterion = CrowdCountLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_mae = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_mae = checkpoint['mae']
        logger.info(f'Resumed from epoch {start_epoch}, best MAE: {best_mae:.2f}')
    
    # Training loop
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    
    logger.info('Starting training...')
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_mae = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        
        logger.info(f'Epoch {epoch+1}/{args.epochs}:')
        logger.info(f'  Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}')
        logger.info(f'  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}')
        
        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_mae,
                os.path.join(args.save_dir, f'{args.model}_best.pth')
            )
            logger.info(f'New best model saved! MAE: {best_mae:.2f}')
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_mae,
                os.path.join(args.save_dir, f'{args.model}_epoch_{epoch+1}.pth')
            )
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_maes, label='Train MAE')
    plt.plot(val_maes, label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Training and Validation MAE')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_curves.png'))
    plt.show()
    
    logger.info(f'Training completed! Best MAE: {best_mae:.2f}')

if __name__ == '__main__':
    main()