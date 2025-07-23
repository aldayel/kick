#!/usr/bin/env python3
"""
Red Dot Detection Demo
Shows how the crowd counting model will visualize detections with red dots
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

def create_demo_image_with_crowd():
    """Create a synthetic image with people for demonstration"""
    # Create a simple scene
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add some simple "people" (circles) for demo
    people_positions = [
        (100, 150), (200, 160), (300, 170), (450, 180),
        (150, 250), (250, 260), (350, 270), (500, 280),
        (120, 350), (220, 360), (320, 370), (420, 380)
    ]
    
    # Draw simple people representations (dark circles)
    for x, y in people_positions:
        cv2.circle(img, (x, y), 15, (50, 50, 50), -1)  # Dark gray circles
    
    return img, people_positions

def add_red_dots_to_detections(image, detections, dot_size=8, confidence_threshold=0.5):
    """
    Add red dots to detected people positions
    
    Args:
        image: Input image (numpy array)
        detections: List of (x, y, confidence) tuples
        dot_size: Size of red dots
        confidence_threshold: Minimum confidence to show detection
    
    Returns:
        Image with red dots overlaid
    """
    result_img = image.copy()
    
    for detection in detections:
        if len(detection) == 3:
            x, y, confidence = detection
            if confidence >= confidence_threshold:
                # Draw red dot
                cv2.circle(result_img, (int(x), int(y)), dot_size, (0, 0, 255), -1)
                # Add white border for better visibility
                cv2.circle(result_img, (int(x), int(y)), dot_size + 1, (255, 255, 255), 2)
        else:
            x, y = detection
            # Draw red dot
            cv2.circle(result_img, (int(x), int(y)), dot_size, (0, 0, 255), -1)
            # Add white border for better visibility
            cv2.circle(result_img, (int(x), int(y)), dot_size + 1, (255, 255, 255), 2)
    
    return result_img

def simulate_crowd_detection(image):
    """
    Simulate crowd detection on an image
    In real implementation, this would use the trained model
    """
    # For demo, we'll use simple color-based detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find dark regions (simulating people)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        # Get contour center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area = cv2.contourArea(contour)
            
            # Filter by size (approximate person size)
            if 200 < area < 2000:
                confidence = min(1.0, area / 1000.0)  # Simulate confidence score
                detections.append((cx, cy, confidence))
    
    return detections

def create_comparison_plot(original, with_dots, detections):
    """Create a side-by-side comparison plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Original Image')
    ax1.axis('off')
    
    # Image with red dots
    ax2.imshow(cv2.cvtColor(with_dots, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Detected People (Red Dots) - Count: {len(detections)}')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

def demo_red_dot_detection(input_path=None, output_path="demo_result.jpg"):
    """Run the red dot detection demo"""
    
    if input_path and os.path.exists(input_path):
        # Load real image
        print(f"Loading image: {input_path}")
        image = cv2.imread(input_path)
        if image is None:
            print("Error: Could not load image")
            return
    else:
        # Create demo image
        print("Creating demo image with synthetic crowd...")
        image, true_positions = create_demo_image_with_crowd()
    
    print("Running crowd detection simulation...")
    detections = simulate_crowd_detection(image)
    
    print(f"Detected {len(detections)} people")
    
    # Add red dots
    print("Adding red dots to detections...")
    result_image = add_red_dots_to_detections(image, detections)
    
    # Create comparison plot
    fig = create_comparison_plot(image, result_image, detections)
    
    # Save results
    cv2.imwrite(output_path, result_image)
    fig.savefig(output_path.replace('.jpg', '_comparison.png'), dpi=150, bbox_inches='tight')
    
    print(f"✓ Results saved:")
    print(f"  - Image with red dots: {output_path}")
    print(f"  - Comparison plot: {output_path.replace('.jpg', '_comparison.png')}")
    
    # Print detection details
    print(f"\nDetection Summary:")
    print(f"Total detections: {len(detections)}")
    for i, detection in enumerate(detections):
        if len(detection) == 3:
            x, y, conf = detection
            print(f"  Person {i+1}: ({x:3.0f}, {y:3.0f}) - Confidence: {conf:.2f}")
        else:
            x, y = detection
            print(f"  Person {i+1}: ({x:3.0f}, {y:3.0f})")
    
    return result_image, detections

def main():
    parser = argparse.ArgumentParser(description="Demo red dot crowd detection")
    parser.add_argument("--input", help="Input image path (optional, will create demo image if not provided)")
    parser.add_argument("--output", default="demo_result.jpg", help="Output image path")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("RED DOT CROWD DETECTION DEMO")
    print("=" * 50)
    
    demo_red_dot_detection(args.input, args.output)
    
    print("\n" + "=" * 50)
    print("DEMO COMPLETE!")
    print("=" * 50)
    print("This demo shows how the trained model will:")
    print("1. Process input images/videos")
    print("2. Detect people locations")
    print("3. Mark each detected person with a red dot")
    print("4. Provide confidence scores for each detection")
    print("\nOnce you train the model with your JHU dataset,")
    print("the actual detection accuracy will be much higher!")

if __name__ == "__main__":
    main()