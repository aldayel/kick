#!/usr/bin/env python3
"""
Quick Start Script for JHU Crowd Analysis
Automates the entire pipeline from dataset setup to training
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dataset_exists():
    """Check if dataset is properly structured"""
    dataset_path = Path("jhu_crowd_v2.0")
    
    print("🔍 Checking dataset structure...")
    
    if not dataset_path.exists():
        print("❌ Dataset directory not found!")
        print("📋 Please:")
        print("   1. Copy your jhu_crowd_v2.0.zip to this directory")
        print("   2. Run: python transfer_dataset.py --zip_path jhu_crowd_v2.0.zip")
        return False
    
    # Check for expected structure
    required_dirs = ["train/images", "train/gt", "val/images", "val/gt", "test/images", "test/gt"]
    missing_dirs = []
    
    for req_dir in required_dirs:
        full_path = dataset_path / req_dir
        if not full_path.exists():
            missing_dirs.append(req_dir)
        else:
            file_count = len(list(full_path.glob('*')))
            print(f"✅ {req_dir}: {file_count} files")
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        print("📋 Run: python prepare_jhu_dataset.py to organize your data")
        return False
    
    print("✅ Dataset structure looks good!")
    return True

def run_command(command, description):
    """Run a command with nice output"""
    print(f"\n🚀 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout[-200:])  # Show last 200 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print("Error details:", e.stderr[-200:])
        return False

def quick_start_training():
    """Start training with recommended settings"""
    
    print("=" * 60)
    print("🎯 QUICK START: JHU CROWD ANALYSIS TRAINING")
    print("=" * 60)
    
    # Step 1: Check dataset
    if not check_dataset_exists():
        print("\n❌ Dataset not ready. Please follow the instructions above.")
        return False
    
    # Step 2: Activate environment and start training
    print("\n🏋️ Starting training with recommended settings...")
    
    # For CPU training (since GPU not available in this environment)
    training_command = (
        "source crowd_env/bin/activate && "
        "python train_crowd_model.py "
        "--data_root ./jhu_crowd_v2.0 "
        "--model csrnet "
        "--batch_size 4 "  # Smaller batch for CPU
        "--epochs 50 "      # Fewer epochs for testing
        "--lr 1e-4 "
        "--device cpu "
        "--save_freq 10"
    )
    
    if not run_command(training_command, "Training CSRNet model"):
        print("❌ Training failed!")
        return False
    
    print("\n🎉 Training completed successfully!")
    
    # Step 3: Test inference
    print("\n🔍 Testing inference with red dot visualization...")
    
    test_command = (
        "source crowd_env/bin/activate && "
        "python inference_with_red_dots.py "
        "--model_path ./checkpoints/csrnet_best.pth "
        "--input demo_result.jpg "
        "--output ./results/test_with_red_dots.jpg"
    )
    
    if run_command(test_command, "Testing inference"):
        print("✅ Inference test successful!")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Quick start for crowd analysis training")
    parser.add_argument("--check-only", action="store_true", help="Only check dataset, don't train")
    parser.add_argument("--demo", action="store_true", help="Run demo first")
    
    args = parser.parse_args()
    
    if args.demo:
        print("🎬 Running red dot detection demo...")
        if run_command("source crowd_env/bin/activate && python demo_red_dots.py", "Running demo"):
            print("✅ Demo completed! Check demo_result.jpg and demo_result_comparison.png")
        return
    
    if args.check_only:
        check_dataset_exists()
        return
    
    # Full quick start
    success = quick_start_training()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 QUICK START COMPLETE!")
        print("=" * 60)
        print("📊 Check these files:")
        print("   - Training logs: ./checkpoints/training.log")
        print("   - Best model: ./checkpoints/csrnet_best.pth")
        print("   - Test results: ./results/test_with_red_dots.jpg")
        print("\n🚀 To run inference on your own images:")
        print("   python inference_with_red_dots.py --input your_image.jpg")
    else:
        print("\n❌ Quick start failed. Please check the error messages above.")
        print("📋 You can also run steps manually:")
        print("   1. python transfer_dataset.py --zip_path your_zip_file.zip")
        print("   2. python prepare_jhu_dataset.py")
        print("   3. python train_crowd_model.py")

if __name__ == "__main__":
    main()