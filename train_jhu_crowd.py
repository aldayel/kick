#!/usr/bin/env python3
"""
Complete JHU Crowd v2.0 Training Script
Single file solution for training crowd counting models with red dot detection capability
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
from torchvision.models import vgg16
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import argparse
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import random
import warnings
from pathlib import Path
import glob

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class CSRNet(nn.Module):
    """CSRNet architecture for crowd counting"""
    
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        
        # Load VGG16 as backbone
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        
        # Frontend (VGG16 first 10 layers)
        self.frontend = self._make_layers(self.frontend_feat)
        
        # Backend (dilated conv layers)
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
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        if load_weights:
            self._initialize_weights()
            
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

class JHUCrowdDataset(Dataset):
    """JHU Crowd v2.0 Dataset Loader"""
    
    def __init__(self, data_path, phase='train', transform=None, target_size=(512, 512)):
        self.data_path = Path(data_path)
        self.phase = phase
        self.transform = transform
        self.target_size = target_size
        
        # Get image and ground truth paths
        self.img_path = self.data_path / phase / 'images'
        self.gt_path = self.data_path / phase / 'gt'
        
        # Get all image files
        self.img_files = sorted(glob.glob(str(self.img_path / '*.jpg')))
        if not self.img_files:
            self.img_files = sorted(glob.glob(str(self.img_path / '*.png')))
        
        logger.info(f"Found {len(self.img_files)} images in {phase} set")
        
        if len(self.img_files) == 0:
            raise FileNotFoundError(f"No images found in {self.img_path}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get corresponding ground truth file
        img_name = Path(img_path).stem
        
        # Try different GT file extensions and formats
        gt_file = None
        for ext in ['.txt', '.mat', '.json']:
            potential_gt = self.gt_path / f"{img_name}{ext}"
            if potential_gt.exists():
                gt_file = potential_gt
                break
        
        if gt_file is None:
            # If no GT file found, create empty annotations
            logger.warning(f"No ground truth found for {img_name}, using empty annotations")
            points = np.array([]).reshape(0, 2)
        else:
            points = self._load_ground_truth(gt_file)
        
        # Original image size
        h, w = image.shape[:2]
        
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Scale points accordingly
        if len(points) > 0:
            points[:, 0] = points[:, 0] * (self.target_size[0] / w)
            points[:, 1] = points[:, 1] * (self.target_size[1] / h)
        
        # Generate density map
        density_map = self._generate_density_map(points, self.target_size)
        
        # Convert to tensors
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        density_map = torch.from_numpy(density_map).float().unsqueeze(0)
        
        return image, density_map, len(points)
    
    def _load_ground_truth(self, gt_file):
        """Load ground truth annotations from various formats"""
        gt_file = Path(gt_file)
        
        if gt_file.suffix == '.txt':
            # Load from text file (x, y coordinates)
            try:
                points = np.loadtxt(gt_file, delimiter=',')
                if points.ndim == 1 and len(points) == 2:
                    points = points.reshape(1, -1)
                elif points.ndim == 1 and len(points) == 0:
                    points = np.array([]).reshape(0, 2)
                return points
            except:
                return np.array([]).reshape(0, 2)
                
        elif gt_file.suffix == '.mat':
            # Load from MATLAB file
            try:
                mat = loadmat(gt_file)
                # Try different possible keys
                for key in ['image_info', 'annPoints', 'points', 'gt']:
                    if key in mat:
                        data = mat[key]
                        if hasattr(data[0][0], 'location'):
                            points = data[0][0].location
                        else:
                            points = data
                        return np.array(points).reshape(-1, 2)
                return np.array([]).reshape(0, 2)
            except:
                return np.array([]).reshape(0, 2)
                
        elif gt_file.suffix == '.json':
            # Load from JSON file
            try:
                with open(gt_file, 'r') as f:
                    data = json.load(f)
                if 'points' in data:
                    return np.array(data['points'])
                elif 'annotations' in data:
                    return np.array(data['annotations'])
                return np.array([]).reshape(0, 2)
            except:
                return np.array([]).reshape(0, 2)
        
        return np.array([]).reshape(0, 2)
    
    def _generate_density_map(self, points, img_size):
        """Generate Gaussian density map from point annotations"""
        h, w = img_size[1], img_size[0]  # Note: img_size is (width, height)
        density_map = np.zeros((h, w), dtype=np.float32)
        
        if len(points) == 0:
            return density_map
        
        # Adaptive sigma based on local density
        for i, point in enumerate(points):
            x, y = int(point[0]), int(point[1])
            
            # Ensure point is within image bounds
            if 0 <= x < w and 0 <= y < h:
                # Calculate adaptive sigma
                if len(points) > 1:
                    distances = np.sqrt(np.sum((points - point) ** 2, axis=1))
                    distances = distances[distances > 0]  # Remove self-distance
                    if len(distances) > 0:
                        sigma = min(distances.mean() / 3, 15)
                    else:
                        sigma = 15
                else:
                    sigma = 15
                
                # Create Gaussian kernel
                sigma = max(sigma, 1.0)  # Minimum sigma
                kernel_size = int(6 * sigma + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                # Create coordinates
                ax = np.arange(kernel_size) - kernel_size // 2
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                kernel = kernel / kernel.sum()
                
                # Place kernel on density map
                y1, y2 = max(0, y - kernel_size//2), min(h, y + kernel_size//2 + 1)
                x1, x2 = max(0, x - kernel_size//2), min(w, x + kernel_size//2 + 1)
                
                ky1, ky2 = max(0, kernel_size//2 - y), kernel_size//2 + (y2 - y1) - (kernel_size//2 + 1 - (y + kernel_size//2 + 1 - y2))
                kx1, kx2 = max(0, kernel_size//2 - x), kernel_size//2 + (x2 - x1) - (kernel_size//2 + 1 - (x + kernel_size//2 + 1 - x2))
                
                if y2 > y1 and x2 > x1:
                    density_map[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]
        
        return density_map

def train_model(data_path, epochs=100, batch_size=4, learning_rate=1e-5, save_dir='./models'):
    """Main training function"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    try:
        train_dataset = JHUCrowdDataset(data_path, phase='train', transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # Try to create validation dataset
        try:
            val_dataset = JHUCrowdDataset(data_path, phase='val', transform=transform)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            logger.info(f"Using separate validation set with {len(val_dataset)} images")
        except:
            # If no val set, use test set or split train set
            try:
                val_dataset = JHUCrowdDataset(data_path, phase='test', transform=transform)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
                logger.info(f"Using test set for validation with {len(val_dataset)} images")
            except:
                # Split training set for validation
                val_size = len(train_dataset) // 5  # 20% for validation
                train_size = len(train_dataset) - val_size
                train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
                logger.info(f"Split training set: {train_size} train, {val_size} validation")
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        logger.error("Please check that your dataset path is correct and contains the proper structure:")
        logger.error("  jhu_crowd_v2.0/")
        logger.error("  ├── train/")
        logger.error("  │   ├── images/")
        logger.error("  │   └── gt/")
        logger.error("  ├── val/ (optional)")
        logger.error("  └── test/ (optional)")
        return
    
    # Create model
    model = CSRNet(load_weights=True).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Training loop
    best_mae = float('inf')
    train_losses = []
    val_maes = []
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        train_mae = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (images, density_maps, counts) in enumerate(train_pbar):
            images = images.to(device)
            density_maps = density_maps.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Resize outputs to match density map size
            outputs = F.interpolate(outputs, size=density_maps.shape[-2:], mode='bilinear', align_corners=False)
            
            # Calculate loss
            loss = criterion(outputs, density_maps)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            epoch_loss += loss.item()
            
            # Calculate MAE
            pred_counts = outputs.sum(dim=(2, 3)).cpu().numpy()
            true_counts = np.array(counts)
            batch_mae = np.mean(np.abs(pred_counts - true_counts))
            train_mae += batch_mae
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MAE': f'{batch_mae:.2f}'
            })
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_mae = 0.0
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for images, density_maps, counts in val_pbar:
                images = images.to(device)
                density_maps = density_maps.to(device)
                
                outputs = model(images)
                outputs = F.interpolate(outputs, size=density_maps.shape[-2:], mode='bilinear', align_corners=False)
                
                loss = criterion(outputs, density_maps)
                val_loss += loss.item()
                
                # Calculate MAE
                pred_counts = outputs.sum(dim=(2, 3)).cpu().numpy()
                true_counts = np.array(counts)
                batch_mae = np.mean(np.abs(pred_counts - true_counts))
                val_mae += batch_mae
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'MAE': f'{batch_mae:.2f}'
                })
        
        avg_val_mae = val_mae / len(val_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_maes.append(avg_val_mae)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Log epoch results
        logger.info(f'Epoch {epoch+1}/{epochs}:')
        logger.info(f'  Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.2f}')
        logger.info(f'  Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.2f}')
        logger.info(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save best model
        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mae': best_mae,
                'train_losses': train_losses,
                'val_maes': val_maes
            }, os.path.join(save_dir, 'best_crowd_model.pth'))
            logger.info(f'  New best model saved! MAE: {best_mae:.2f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_maes': val_maes
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_maes': val_maes
    }, os.path.join(save_dir, 'final_crowd_model.pth'))
    
    logger.info(f'Training completed! Best validation MAE: {best_mae:.2f}')
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_maes)
    plt.title('Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.show()
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train JHU Crowd v2.0 Model')
    parser.add_argument('--data_path', type=str, 
                       default=r'C:\Users\user\OneDrive\Desktop\makkah-crowd-analysis\jhu_crowd_v2.0',
                       help='Path to JHU Crowd v2.0 dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Convert Windows path to current system format if needed
    if os.name != 'nt' and '\\' in args.data_path:
        logger.info("Converting Windows path format...")
        # For now, assume dataset is in current directory
        args.data_path = './jhu_crowd_v2.0'
    
    logger.info(f"Dataset path: {args.data_path}")
    logger.info(f"Training parameters:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Save directory: {args.save_dir}")
    
    # Check if dataset exists
    if not os.path.exists(args.data_path):
        logger.error(f"Dataset path does not exist: {args.data_path}")
        logger.error("Please ensure the JHU Crowd v2.0 dataset is available at the specified path.")
        logger.error("Expected structure:")
        logger.error("  jhu_crowd_v2.0/")
        logger.error("  ├── train/")
        logger.error("  │   ├── images/")
        logger.error("  │   └── gt/")
        logger.error("  ├── val/ (optional)")
        logger.error("  └── test/ (optional)")
        return
    
    # Start training
    model = train_model(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Models saved in: {args.save_dir}")
    logger.info("You can now use the trained model for inference with red dot detection.")

if __name__ == '__main__':
    main()