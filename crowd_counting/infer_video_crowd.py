import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, maximum_filter

from train_jhu_crowd import BasicCSRNet  # reuse model definition


NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect_peaks(density_map, thresh=0.1, min_distance=4):
    """Return (x, y) coordinates of peaks in the density map."""
    # Smooth to reduce noise
    density_blur = gaussian_filter(density_map, sigma=1)
    # Non-maximum suppression
    max_f = maximum_filter(density_blur, size=min_distance * 2 + 1)
    peaks = (density_blur == max_f) & (density_blur > thresh)
    ys, xs = np.where(peaks)
    return list(zip(xs, ys))


def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = transforms.ToTensor()(frame_rgb)
    img = NORMALIZE(img)
    return img.unsqueeze(0)  # (1, 3, H, W)


def main():
    parser = argparse.ArgumentParser(description="Run crowd counting model on a video and draw red dots for detections.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input MP4 video")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model .pth file")
    parser.add_argument("--output", type=str, default="annotated_output.mp4", help="Path to save annotated video")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--thresh", type=float, default=0.1, help="Peak detection threshold (density value)")
    parser.add_argument("--min_distance", type=int, default=4, help="Minimum distance between peaks in pixels (on upsampled map)")
    parser.add_argument("--downsample", type=int, default=8, help="Downsample factor used during training")
    args = parser.parse_args()

    assert Path(args.video_path).exists(), "Video path does not exist"
    assert Path(args.checkpoint).exists(), "Checkpoint not found"

    model = BasicCSRNet().to(args.device)
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    with torch.no_grad():
        for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
            img_tensor = preprocess_frame(frame).to(args.device)
            pred = model(img_tensor)
            pred_np = pred.squeeze().cpu().numpy()

            # Upsample to original resolution and scale back
            density_up = cv2.resize(pred_np, (width, height), interpolation=cv2.INTER_CUBIC) / (args.downsample ** 2)

            # Detect peaks and draw red dots
            peaks = detect_peaks(density_up, thresh=args.thresh, min_distance=args.min_distance)
            for x, y in peaks:
                cv2.circle(frame, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)

            # Optionally overlay count text
            count = int(density_up.sum())
            cv2.putText(frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            writer.write(frame)

    cap.release()
    writer.release()
    print(f"Annotated video saved to {args.output}")


if __name__ == "__main__":
    main()