#!/usr/bin/env python3
"""
Setup script for JHU Crowd Analysis Pipeline
Installs dependencies and verifies environment
"""

import subprocess
import sys
import os
import torch

def run_command(command):
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {command}")
        print(f"Error: {e.stderr}")
        return False

def check_gpu():
    """Check GPU availability and CUDA support"""
    print("\n=== GPU Check ===")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        print(f"✓ GPU count: {gpu_count}")
        print(f"✓ GPU name: {gpu_name}")
        print(f"✓ GPU memory: {gpu_memory:.1f} GB")
        
        # Recommend batch size based on GPU memory
        if gpu_memory >= 20:
            recommended_batch = 16
        elif gpu_memory >= 12:
            recommended_batch = 8
        elif gpu_memory >= 8:
            recommended_batch = 4
        else:
            recommended_batch = 2
            
        print(f"✓ Recommended batch size: {recommended_batch}")
        return True
    else:
        print("✗ CUDA not available - will use CPU (slower)")
        return False

def install_dependencies():
    """Install required Python packages"""
    print("\n=== Installing Dependencies ===")
    
    # Install from requirements.txt
    if os.path.exists('requirements.txt'):
        success = run_command(f"{sys.executable} -m pip install -r requirements.txt")
        if not success:
            print("Failed to install from requirements.txt")
            return False
    else:
        # Install packages individually
        packages = [
            'torch>=1.12.0',
            'torchvision>=0.13.0',
            'opencv-python>=4.6.0',
            'numpy>=1.21.0',
            'scipy>=1.8.0',
            'matplotlib>=3.5.0',
            'Pillow>=8.3.0',
            'tqdm>=4.64.0',
            'scikit-learn>=1.1.0'
        ]
        
        for package in packages:
            success = run_command(f"{sys.executable} -m pip install {package}")
            if not success:
                print(f"Failed to install {package}")
                return False
    
    return True

def verify_imports():
    """Verify that all required packages can be imported"""
    print("\n=== Verifying Imports ===")
    
    required_packages = [
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'scipy',
        'matplotlib',
        'PIL',
        'tqdm',
        'sklearn'
    ]
    
    all_success = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            all_success = False
    
    return all_success

def create_directories():
    """Create necessary directories"""
    print("\n=== Creating Directories ===")
    
    directories = [
        'checkpoints',
        'jhu_crowd_v2.0',
        'results'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Failed to create {directory}: {e}")
            return False
    
    return True

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*50)
    print("SETUP COMPLETE! Next Steps:")
    print("="*50)
    print()
    print("1. Transfer your JHU Crowd v2.0 dataset to this workspace:")
    print("   - Copy the dataset to ./jhu_crowd_v2.0/")
    print()
    print("2. Organize the dataset (if needed):")
    print("   python prepare_jhu_dataset.py \\")
    print("       --source ./jhu_crowd_v2.0 \\")
    print("       --target ./jhu_crowd_v2.0_organized")
    print()
    print("3. Start training:")
    print("   python train_crowd_model.py \\")
    print("       --data_root ./jhu_crowd_v2.0_organized \\")
    print("       --model csrnet \\")
    print("       --batch_size 8 \\")
    print("       --epochs 100")
    print()
    print("4. Run inference with red dots:")
    print("   python inference_with_red_dots.py \\")
    print("       --model_path ./checkpoints/csrnet_best.pth \\")
    print("       --input /path/to/image_or_video \\")
    print("       --output ./result_with_red_dots.jpg")
    print()
    print("See README.md for detailed instructions!")

def main():
    """Main setup function"""
    print("JHU Crowd Analysis Pipeline Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("✗ Python 3.7+ required")
        sys.exit(1)
    else:
        print(f"✓ Python {sys.version}")
    
    # Install dependencies
    if not install_dependencies():
        print("✗ Failed to install dependencies")
        sys.exit(1)
    
    # Verify imports
    if not verify_imports():
        print("✗ Failed to import required packages")
        sys.exit(1)
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Create directories
    if not create_directories():
        print("✗ Failed to create directories")
        sys.exit(1)
    
    print("\n✓ Setup completed successfully!")
    
    if not gpu_available:
        print("\nWARNING: No GPU detected. Training will be slower on CPU.")
        print("Consider using a machine with CUDA-compatible GPU for better performance.")
    
    print_next_steps()

if __name__ == '__main__':
    main()