#!/usr/bin/env python3
"""
JHU Crowd Model Inference with Red Dot Visualization
Processes images/videos and shows detected people with red dots
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.ndimage import maximum_filter
from scipy.ndimage import label
import argparse
import logging
from PIL import Image
import matplotlib.pyplot as plt
from train_crowd_model import CSRNet, CANNet

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrowdDetector:
    """Crowd detection with red dot visualization"""
    
    def __init__(self, model_path, model_type='csrnet', device='cuda', confidence_threshold=0.1):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Load model
        if model_type == 'csrnet':
            self.model = CSRNet(pretrained=False)
        elif model_type == 'cannet':
            self.model = CANNet(pretrained=False)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Loaded {model_type} model on {self.device}")
    
    def detect_peaks(self, density_map, min_distance=10, threshold_abs=None):
        """
        Detect local maxima in density map to find individual person locations
        """
        if threshold_abs is None:
            threshold_abs = self.confidence_threshold * np.max(density_map)
        
        # Apply maximum filter to find local maxima
        footprint = np.ones((min_distance, min_distance))
        local_maxima = maximum_filter(density_map, footprint=footprint) == density_map
        
        # Apply threshold
        above_threshold = density_map > threshold_abs
        
        # Combine conditions
        peaks = local_maxima & above_threshold
        
        # Get peak coordinates
        peak_coords = np.where(peaks)
        peak_positions = list(zip(peak_coords[1], peak_coords[0]))  # (x, y) format
        peak_values = density_map[peak_coords]
        
        return peak_positions, peak_values
    
    def predict_density_map(self, image):
        """
        Predict density map for input image
        """
        original_shape = image.shape[:2]
        
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            density_pred = self.model(input_tensor)
            
            # Resize back to original image size
            density_pred = F.interpolate(
                density_pred, 
                size=original_shape, 
                mode='bilinear', 
                align_corners=False
            )
            
            # Convert to numpy
            density_map = density_pred.squeeze().cpu().numpy()
            
            # Ensure non-negative values
            density_map = np.maximum(density_map, 0)
        
        return density_map
    
    def detect_people(self, image_path, output_path=None, show_density=False):
        """
        Detect people in image and visualize with red dots
        """
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_path
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        original_height, original_width = image_rgb.shape[:2]
        
        # Predict density map
        density_map = self.predict_density_map(image_rgb)
        
        # Detect individual people locations
        peak_positions, peak_values = self.detect_peaks(density_map)
        
        # Count people
        people_count = len(peak_positions)
        total_density = np.sum(density_map)
        
        logger.info(f"Detected {people_count} people (Total density: {total_density:.2f})")
        
        # Create visualization
        result_image = image_rgb.copy()
        
        # Draw red dots for each detected person
        dot_radius = max(3, min(original_width, original_height) // 200)
        
        for (x, y), confidence in zip(peak_positions, peak_values):
            # Scale dot size based on confidence
            radius = max(2, int(dot_radius * (confidence / np.max(peak_values))))
            
            # Draw red dot
            cv2.circle(result_image, (int(x), int(y)), radius, (255, 0, 0), -1)
            
            # Optional: Draw confidence value
            if confidence > self.confidence_threshold * 2:
                cv2.putText(result_image, f'{confidence:.2f}', 
                           (int(x) + radius + 2, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # Add count text
        count_text = f"People Count: {people_count}"
        cv2.putText(result_image, count_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Create visualization with density map if requested
        if show_density:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image_rgb)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Density map
            density_viz = axes[1].imshow(density_map, cmap='hot', alpha=0.8)
            axes[1].set_title(f'Density Map (Sum: {total_density:.2f})')
            axes[1].axis('off')
            plt.colorbar(density_viz, ax=axes[1])
            
            # Result with red dots
            axes[2].imshow(result_image)
            axes[2].set_title(f'Detections: {people_count} people')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path.replace('.jpg', '_analysis.png'), 
                           dpi=150, bbox_inches='tight')
            plt.show()
        
        # Save result image
        if output_path:
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
            logger.info(f"Saved result to {output_path}")
        
        return result_image, people_count, peak_positions, density_map
    
    def process_video(self, video_path, output_path, skip_frames=1):
        """
        Process video and add red dots for crowd detection
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_frames = 0
        
        logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for faster processing
            if frame_count % skip_frames == 0:
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect people
                result_frame, count, positions, _ = self.detect_people(frame_rgb)
                
                # Convert back to BGR
                result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(result_frame)
                processed_frames += 1
                
                if processed_frames % 30 == 0:
                    logger.info(f"Processed {processed_frames} frames, Current count: {count}")
            else:
                # Write original frame
                out.write(frame)
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        out.release()
        
        logger.info(f"Video processing complete: {output_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Crowd Detection with Red Dots')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='csrnet', 
                      choices=['csrnet', 'cannet'],
                      help='Model architecture type')
    parser.add_argument('--input', type=str, required=True,
                      help='Input image or video path')
    parser.add_argument('--output', type=str, required=True,
                      help='Output path for result')
    parser.add_argument('--confidence_threshold', type=float, default=0.1,
                      help='Confidence threshold for detection')
    parser.add_argument('--show_density', action='store_true',
                      help='Show density map visualization')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--skip_frames', type=int, default=1,
                      help='Skip frames for video processing (1=process all)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = CrowdDetector(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )
    
    # Check if input is video or image
    input_ext = os.path.splitext(args.input)[1].lower()
    
    if input_ext in ['.mp4', '.avi', '.mov', '.mkv']:
        # Process video
        detector.process_video(args.input, args.output, args.skip_frames)
    else:
        # Process image
        detector.detect_people(args.input, args.output, args.show_density)

if __name__ == '__main__':
    main()