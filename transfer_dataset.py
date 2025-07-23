#!/usr/bin/env python3
"""
Dataset Transfer and Verification Script
Helps transfer JHU Crowd v2.0 dataset and verify its structure
"""

import os
import shutil
import zipfile
from pathlib import Path
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_zip_dataset(zip_path, extract_to):
    """Extract JHU Crowd v2.0 zip file"""
    logger.info(f"Extracting {zip_path} to {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    logger.info("Extraction completed!")

def verify_dataset_structure(dataset_path):
    """Verify the JHU Crowd v2.0 dataset structure"""
    dataset_path = Path(dataset_path)
    
    logger.info(f"Verifying dataset structure at: {dataset_path}")
    
    # Check if main directory exists
    if not dataset_path.exists():
        logger.error(f"Dataset directory does not exist: {dataset_path}")
        return False
    
    # Expected structure patterns
    expected_dirs = ['train', 'val', 'test']
    expected_subdirs = ['images', 'gt']
    
    structure_valid = True
    
    # Check for standard structure
    for split in expected_dirs:
        split_dir = dataset_path / split
        if split_dir.exists():
            logger.info(f"✓ Found {split} directory")
            
            for subdir in expected_subdirs:
                subdir_path = split_dir / subdir
                if subdir_path.exists():
                    count = len(list(subdir_path.glob('*')))
                    logger.info(f"  ✓ {split}/{subdir}: {count} files")
                else:
                    logger.warning(f"  ✗ Missing {split}/{subdir}")
        else:
            logger.warning(f"✗ Missing {split} directory")
    
    # Also check for flat structure
    image_files = list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.png'))
    gt_files = list(dataset_path.glob('*.txt')) + list(dataset_path.glob('*.mat'))
    
    if image_files:
        logger.info(f"Found {len(image_files)} image files in root directory")
    if gt_files:
        logger.info(f"Found {len(gt_files)} ground truth files in root directory")
    
    return True

def create_sample_structure():
    """Create a sample dataset structure for demonstration"""
    logger.info("Creating sample dataset structure...")
    
    base_dir = Path("jhu_crowd_v2.0")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'gt']:
            dir_path = base_dir / split / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create README
    readme_content = """
JHU Crowd v2.0 Dataset Structure
================================

Please place your dataset files in the following structure:

jhu_crowd_v2.0/
├── train/
│   ├── images/          # Training images (.jpg, .png)
│   └── gt/             # Ground truth files (.txt with x,y coordinates)
├── val/
│   ├── images/          # Validation images
│   └── gt/             # Ground truth files
└── test/
    ├── images/          # Test images
    └── gt/             # Ground truth files

Ground Truth Format:
Each .txt file should contain one line per person with x,y coordinates:
x1,y1
x2,y2
...

Example:
245.5,123.2
156.8,234.7
"""
    
    with open(base_dir / "README_DATASET.txt", "w") as f:
        f.write(readme_content)
    
    logger.info(f"✓ Created sample structure at: {base_dir}")

def main():
    parser = argparse.ArgumentParser(description="Transfer and verify JHU Crowd v2.0 dataset")
    parser.add_argument("--zip_path", help="Path to jhu_crowd_v2.0.zip file")
    parser.add_argument("--dataset_path", default="jhu_crowd_v2.0", help="Path to dataset directory")
    parser.add_argument("--extract_to", default=".", help="Directory to extract to")
    parser.add_argument("--create_sample", action="store_true", help="Create sample directory structure")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_structure()
        return
    
    if args.zip_path:
        if os.path.exists(args.zip_path):
            extract_zip_dataset(args.zip_path, args.extract_to)
        else:
            logger.error(f"Zip file not found: {args.zip_path}")
            return
    
    # Verify dataset structure
    verify_dataset_structure(args.dataset_path)
    
    print("\n" + "="*50)
    print("DATASET TRANSFER INSTRUCTIONS")
    print("="*50)
    print("1. Copy your jhu_crowd_v2.0.zip to this workspace")
    print("2. Run: python transfer_dataset.py --zip_path jhu_crowd_v2.0.zip")
    print("3. Or manually copy extracted files to ./jhu_crowd_v2.0/")
    print("4. Run: python prepare_jhu_dataset.py to organize the data")
    print("5. Start training: python train_crowd_model.py")

if __name__ == "__main__":
    main()