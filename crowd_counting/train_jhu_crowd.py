import os
import argparse
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import scipy.io as sio
import cv2


def generate_density_map(image, points, sigma=4):
    """Generate a simple Gaussian density map.

    Args:
        image (np.ndarray): Input image array (H, W, 3).
        points (np.ndarray): Nx2 array of point coordinates (x, y).
        sigma (float): Standard deviation for the Gaussian kernel.

    Returns:
        np.ndarray: Density map of shape (H, W).
    """
    h, w = image.shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    for p in points:
        x, y = min(int(p[0]), w - 1), min(int(p[1]), h - 1)
        density[y, x] += 1

    density = cv2.GaussianBlur(density, (0, 0), sigma)

    return density


class JHUCrowdDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, downsample=8):
        """PyTorch dataset for JHU-CROWD v2.0.

        Args:
            root_dir (str): Path to dataset root containing "train", "val", "test" directories.
            split (str): One of {"train", "val", "test"}.
            transform (callable, optional): Transform applied to the PIL Image.
            downsample (int): Downsample factor between image and density map (e.g., 8 for CSRNet).
        """
        assert split in {"train", "val", "test"}
        self.image_paths = sorted(glob(os.path.join(root_dir, split, "*.jpg")))
        self.transform = transform
        self.downsample = downsample

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mat_path = img_path.replace(".jpg", ".mat").replace(".png", ".mat")

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        # Load annotations
        mat = sio.loadmat(mat_path)
        # In JHU dataset, annotation is stored as 'image_info' and points in 'location'
        points = mat.get("image_info", mat).get("location", mat["image_info"])[0][0][0]

        # Create density map
        density = generate_density_map(img_np, points)

        if self.transform is not None:
            img = self.transform(img)

        # Downsample density for model output
        ds_rows = img.shape[1] // self.downsample
        ds_cols = img.shape[2] // self.downsample
        density_ds = cv2.resize(density, (ds_cols, ds_rows)) * (self.downsample ** 2)
        density_ds = torch.from_numpy(density_ds).unsqueeze(0)  # (1, H, W)

        return img, density_ds


class BasicCSRNet(nn.Module):
    """Simplified CSRNet architecture suitable for crowd counting."""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16_bn
        vgg = vgg16_bn(pretrained=True)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:33])  # up to conv4_3

        backend_channels = [512, 512, 512, 256, 128, 64]
        layers = []
        in_ch = 512
        for ch in backend_channels:
            layers.append(nn.Conv2d(in_ch, ch, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = ch
        self.backend = nn.Sequential(*layers)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for imgs, densities in loader:
        imgs = imgs.to(device)
        densities = densities.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, densities)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * imgs.size(0)
    return epoch_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    mse, mae = 0.0, 0.0
    with torch.no_grad():
        for imgs, densities in loader:
            imgs = imgs.to(device)
            densities = densities.to(device)
            outputs = model(imgs)
            mse += ((outputs - densities) ** 2).sum().item()
            mae += torch.abs(outputs.sum() - densities.sum()).item()
    mse /= len(loader.dataset)
    mae /= len(loader.dataset)
    return mse ** 0.5, mae


def main():
    parser = argparse.ArgumentParser(description="Train CSRNet on JHU-CROWD v2.0")
    parser.add_argument("--data_root", type=str, required=True, help="Path to JHU-CROWD dataset root")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = JHUCrowdDataset(args.data_root, split="train", transform=transform)
    val_dataset = JHUCrowdDataset(args.data_root, split="val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = BasicCSRNet().to(args.device)

    criterion = nn.MSELoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
        rmse, mae = validate(model, val_loader, criterion, args.device)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val RMSE={rmse:.2f} | Val MAE={mae:.2f}")

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))

    print("Training complete. Best MAE:", best_mae)


if __name__ == "__main__":
    main()