"""Main entry point for the fall detection system."""

import argparse
import logging
import sys
from pathlib import Path
import cv2
import time

from .oopsie_controller import OopsieController

def on_algorithm_fall(frame, landmarks, timestamp):
    """Handle algorithm-detected falls.
    
    Args:
        frame: The frame where fall was detected
        landmarks: The pose landmarks
        timestamp: Time of detection
    """
    # Save the frame with timestamp
    timestamp_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamp))
    output_path = f"algorithm_fall_{timestamp_str}.jpg"
    ## cv2.imwrite(output_path, frame)
    
    logging.warning(f"⚠️  Algorithm detected potential fall at {timestamp_str}")

def on_confirmed_fall(sequence_path, analysis, timestamp):
    """Handle LLM-confirmed falls.
    
    Args:
        sequence_path: Path to the frame sequence image
        analysis: The LLM analysis text
        timestamp: Time of confirmation
    """
    timestamp_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamp))
    
    # Save the sequence with timestamp
    output_path = f"confirmed_fall_{timestamp_str}.jpg"
    cv2.imwrite(output_path, cv2.imread(sequence_path))
    
    # Save the analysis
    analysis_path = f"fall_analysis_{timestamp_str}.txt"
    with open(analysis_path, "w") as f:
        f.write(analysis)
    
    logging.error(f"🚨 LLM CONFIRMED FALL at {timestamp_str}")
    logging.info(f"Sequence saved to: {output_path}")
    logging.info(f"Analysis saved to: {analysis_path}")

def process_video(controller: OopsieController, video_path: str) -> None:
    """Process a video file for fall detection.
    
    Args:
        controller: The OopsieController instance
        video_path: Path to the video file to process
    """
    print(f"\nProcessing video: {video_path}")
    print("Press 'q' to quit video processing")
    
    try:
        controller.process_video(video_path)
    except Exception as e:
        print(f"Error processing video: {str(e)}")

def main() -> None:
    """Main entry point for the fall detection system."""
    parser = argparse.ArgumentParser(description="Fall detection system")
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file to process"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Create controller
    controller = OopsieController()
    
    # Add fall detection subscribers
    controller.add_algorithm_fall_subscriber(on_algorithm_fall)
    controller.add_confirmed_fall_subscriber(on_confirmed_fall)
    
    # Process video if provided
    if args.video:
        process_video(controller, args.video)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 