#!/usr/bin/env python3
"""
JHU Crowd v2.0 Dataset Preparation Script
Organizes the dataset into proper train/val/test splits
"""

import os
import shutil
import json
import random
from pathlib import Path
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def organize_jhu_dataset(source_dir, target_dir, train_split=0.7, val_split=0.15, test_split=0.15):
    """
    Organize JHU Crowd v2.0 dataset into proper directory structure
    
    Expected source structure:
    jhu_crowd_v2.0/
    ├── train/
    │   ├── images/
    │   └── gt/
    ├── val/
    │   ├── images/
    │   └── gt/
    └── test/
        ├── images/
        └── gt/
    
    Or flat structure that needs to be organized.
    """
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory structure
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'gt']:
            (target_path / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Check if source already has proper structure
    if (source_path / 'train').exists() and (source_path / 'val').exists():
        logger.info("Source has proper structure, copying files...")
        
        # Copy existing structure
        for split in ['train', 'val', 'test']:
            source_split = source_path / split
            target_split = target_path / split
            
            if source_split.exists():
                # Copy images
                source_images = source_split / 'images'
                target_images = target_split / 'images'
                
                if source_images.exists():
                    for img_file in source_images.glob('*'):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            shutil.copy2(img_file, target_images / img_file.name)
                
                # Copy ground truth
                source_gt = source_split / 'gt'
                target_gt = target_split / 'gt'
                
                if source_gt.exists():
                    for gt_file in source_gt.glob('*.txt'):
                        shutil.copy2(gt_file, target_gt / gt_file.name)
        
    else:
        # Need to organize flat structure
        logger.info("Organizing flat directory structure...")
        
        # Find all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(source_path.glob(f'**/{ext}'))
        
        # Filter to only files that have corresponding ground truth
        valid_files = []
        for img_file in image_files:
            gt_file = img_file.with_suffix('.txt')
            if not gt_file.exists():
                # Try different naming conventions
                gt_file = source_path / 'gt' / f"{img_file.stem}.txt"
                if not gt_file.exists():
                    gt_file = img_file.parent / 'gt' / f"{img_file.stem}.txt"
            
            if gt_file.exists():
                valid_files.append((img_file, gt_file))
        
        logger.info(f"Found {len(valid_files)} valid image-annotation pairs")
        
        # Shuffle and split
        random.shuffle(valid_files)
        
        n_total = len(valid_files)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_files = valid_files[:n_train]
        val_files = valid_files[n_train:n_train + n_val]
        test_files = valid_files[n_train + n_val:]
        
        # Copy files to appropriate splits
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, file_list in splits.items():
            logger.info(f"Copying {len(file_list)} files to {split_name} split")
            
            for img_file, gt_file in file_list:
                # Copy image
                target_img = target_path / split_name / 'images' / img_file.name
                shutil.copy2(img_file, target_img)
                
                # Copy ground truth
                target_gt = target_path / split_name / 'gt' / f"{img_file.stem}.txt"
                shutil.copy2(gt_file, target_gt)
    
    # Generate dataset statistics
    generate_dataset_stats(target_path)
    
    logger.info(f"Dataset organized successfully in {target_path}")

def generate_dataset_stats(dataset_path):
    """Generate and save dataset statistics"""
    
    stats = {}
    
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if not split_path.exists():
            continue
            
        images_path = split_path / 'images'
        gt_path = split_path / 'gt'
        
        # Count files
        n_images = len(list(images_path.glob('*'))) if images_path.exists() else 0
        n_annotations = len(list(gt_path.glob('*.txt'))) if gt_path.exists() else 0
        
        # Count total people
        total_people = 0
        people_per_image = []
        
        if gt_path.exists():
            for gt_file in gt_path.glob('*.txt'):
                try:
                    with open(gt_file, 'r') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        count = len(lines)
                        total_people += count
                        people_per_image.append(count)
                except Exception as e:
                    logger.warning(f"Error reading {gt_file}: {e}")
        
        stats[split] = {
            'n_images': n_images,
            'n_annotations': n_annotations,
            'total_people': total_people,
            'avg_people_per_image': total_people / max(n_images, 1),
            'min_people': min(people_per_image) if people_per_image else 0,
            'max_people': max(people_per_image) if people_per_image else 0
        }
        
        logger.info(f"{split.upper()} split:")
        logger.info(f"  Images: {n_images}")
        logger.info(f"  Annotations: {n_annotations}")
        logger.info(f"  Total people: {total_people}")
        logger.info(f"  Avg people/image: {stats[split]['avg_people_per_image']:.2f}")
        logger.info(f"  People range: {stats[split]['min_people']}-{stats[split]['max_people']}")
    
    # Save stats to JSON
    with open(dataset_path / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

def verify_dataset(dataset_path):
    """Verify dataset integrity"""
    
    dataset_path = Path(dataset_path)
    issues = []
    
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if not split_path.exists():
            issues.append(f"Missing {split} directory")
            continue
        
        images_path = split_path / 'images'
        gt_path = split_path / 'gt'
        
        if not images_path.exists():
            issues.append(f"Missing {split}/images directory")
            continue
        
        if not gt_path.exists():
            issues.append(f"Missing {split}/gt directory")
            continue
        
        # Check for matching image-annotation pairs
        image_files = {f.stem for f in images_path.glob('*') 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
        gt_files = {f.stem for f in gt_path.glob('*.txt')}
        
        missing_gt = image_files - gt_files
        missing_images = gt_files - image_files
        
        if missing_gt:
            issues.append(f"{split}: {len(missing_gt)} images missing ground truth")
        
        if missing_images:
            issues.append(f"{split}: {len(missing_images)} ground truth files missing images")
    
    if issues:
        logger.warning("Dataset verification issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    else:
        logger.info("Dataset verification passed!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Prepare JHU Crowd v2.0 Dataset')
    parser.add_argument('--source', type=str, required=True,
                      help='Path to source JHU Crowd v2.0 directory')
    parser.add_argument('--target', type=str, required=True,
                      help='Path to target organized dataset directory')
    parser.add_argument('--train_split', type=float, default=0.7,
                      help='Training split ratio')
    parser.add_argument('--val_split', type=float, default=0.15,
                      help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.15,
                      help='Test split ratio')
    parser.add_argument('--verify_only', action='store_true',
                      help='Only verify existing dataset structure')
    
    args = parser.parse_args()
    
    # Validate split ratios
    if abs(args.train_split + args.val_split + args.test_split - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    if args.verify_only:
        verify_dataset(args.target)
    else:
        organize_jhu_dataset(
            args.source, 
            args.target, 
            args.train_split, 
            args.val_split, 
            args.test_split
        )
        verify_dataset(args.target)

if __name__ == '__main__':
    main()