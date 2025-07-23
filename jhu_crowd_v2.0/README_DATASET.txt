
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
