"""Test implementation for the OopsieController system.

This script demonstrates the usage of OopsieController for both image and video processing.
It provides examples of:
- Single image fall detection
- Video stream fall detection
- Error handling and user feedback
"""

import os
import sys
import argparse
from pathlib import Path
import cv2

from elder_care_vision.oopsie_controller.oopsie_controller import OopsieController

def process_image(controller: OopsieController, image_path: str) -> None:
    """Process a single image for fall detection.
    
    Args:
        controller: The OopsieController instance
        image_path: Path to the image file
    """
    try:
        print(f"\nProcessing image: {image_path}")
        processed_frame, fall_detected = controller.process_image(image_path)
        
        if fall_detected:
            print("⚠️  FALL DETECTED! Emergency response may be required.")
        else:
            print("✅ No fall detected - person appears safe.")
            
        # Save processed image
        output_path = Path(image_path).with_stem(f"{Path(image_path).stem}_processed")
        cv2.imwrite(str(output_path), processed_frame)
        print(f"Processed image saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

def process_video(controller: OopsieController, video_path: str) -> None:
    """Process a video file for fall detection.
    
    Args:
        controller: The OopsieController instance
        video_path: Path to the video file
    """
    try:
        print(f"\nProcessing video: {video_path}")
        print("Press 'q' to quit video processing")
        controller.process_video(video_path)
        print("Video processing completed")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")

def main():
    """Main function to test OopsieController implementation."""
    parser = argparse.ArgumentParser(description="Test OopsieController fall detection system")
    parser.add_argument("--image", help="Path to image file for testing")
    parser.add_argument("--video", help="Path to video file for testing")
    args = parser.parse_args()
    
    if not args.image and not args.video:
        print("Error: Please provide either --image or --video argument")
        parser.print_help()
        sys.exit(1)
        
    # Initialize controller
    controller = OopsieController()
    
    # Process image if provided
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            sys.exit(1)
        process_image(controller, args.image)
        
    # Process video if provided
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        process_video(controller, args.video)

if __name__ == "__main__":
    main() 