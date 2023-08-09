from torchvision.transforms import ToPILImage
from torchvision.io import read_image
import argparse
import os
import cv2
import numpy as np
import glob
from pathlib import Path

def create_video_from_image_dir(image_dir: str, fps: int, output_dir: str):
    """Create video from image dir."""
    os.makedirs(output_dir, exist_ok=True)
    video_name = f"{'_'.join(os.path.split(image_dir))}.mp4"
    video_path = os.path.join(output_dir, video_name)
    filenames = glob.glob(os.path.join(image_dir, "*"))
    filenames = sorted(filenames, key=lambda x: Path(x).stem)
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    width, height = ToPILImage()(read_image(filenames[0])).size
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for filename in filenames:
        image_data = ToPILImage()(read_image(filename))
        frame = np.array(image_data)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()
    return

def get_arguments():
    """Get arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', help="specify directory containing images", type=str, required=True)
    parser.add_argument("--fps", help="specify frames per second", type=int, default=10)
    parser.add_argument("--output_dir", help="specify output directory.", type=str, default="video_visualizations")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    """Main function."""
    args = get_arguments()
    create_video_from_image_dir(image_dir=args.image_dir, fps=args.fps, output_dir=args.output_dir)
    exit(0)