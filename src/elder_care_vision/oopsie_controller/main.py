"""Main entry point for the fall detection system."""

import argparse
import logging
import sys
from pathlib import Path
import cv2
import time
import numpy as np

from .oopsie_controller import OopsieController

def create_frame_sequence(frames, timestamps, width=None):
    """Create a horizontal sequence of frames.
    
    Args:
        frames: List of frames to combine
        timestamps: List of timestamps for each frame
        width: Optional width to resize frames to
    
    Returns:
        Combined image showing frame sequence
    """
    if not frames:
        return None
        
    # If width specified, resize all frames
    if width:
        resized_frames = []
        for frame in frames:
            height = int(frame.shape[0] * (width / frame.shape[1]))
            resized_frames.append(cv2.resize(frame, (width, height)))
        frames = resized_frames
    
    # Calculate dimensions for the sequence image
    frame_height, frame_width = frames[0].shape[:2]
    sequence_width = frame_width * 3  # 3 frames per row
    sequence_height = frame_height * 2  # 2 rows
    
    # Create blank image for the sequence
    sequence = np.zeros((sequence_height, sequence_width, 3), dtype=np.uint8)
    
    # Add frames to sequence with timestamps
    latest_time = timestamps[-1]
    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        row = i // 3
        col = i % 3
        y_start = row * frame_height
        x_start = col * frame_width
        
        # Add frame to sequence
        sequence[y_start:y_start + frame_height, x_start:x_start + frame_width] = frame
        
        # Add timestamp with time difference
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        time_diff = latest_time - timestamp
        timestamp_str = f"t-{time_diff:.1f}s"
        cv2.putText(sequence, timestamp_str,
                   (x_start + 10, y_start + frame_height - 10),
                   font, font_scale, (255, 255, 255), thickness)
    
    return sequence

def on_algorithm_fall(frame, landmarks, timestamp):
    """Handle algorithm-detected falls.
    
    Args:
        frame: The frame where fall was detected
        landmarks: The pose landmarks
        timestamp: Time of detection
    """
    timestamp_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamp))
    output_path = f"algorithm_fall_{timestamp_str}.jpg"
    ## cv2.imwrite(output_path, frame)
    
    logging.warning(f"⚠️  Algorithm detected potential fall at {timestamp_str}")

def on_confirmed_fall(sequence_frames, sequence_timestamps, analysis, timestamp):
    """Handle LLM-confirmed falls.
    
    Args:
        sequence_frames: List of frames showing the fall sequence
        sequence_timestamps: List of timestamps for each frame
        analysis: The LLM analysis text
        timestamp: Time of confirmation
    """
    if not sequence_frames or not sequence_timestamps:
        logging.error("No frames or timestamps provided for confirmed fall")
        return
        
    timestamp_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamp))
    
    try:
        # Create sequence image from frames
        sequence = create_frame_sequence(sequence_frames, sequence_timestamps, width=400)
        if sequence is None:
            logging.error("Failed to create frame sequence")
            return
            
        # Add analysis text to the bottom of the sequence
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        padding = 20
        
        # Add padding at the bottom for text
        text_height = 100  # Fixed height for text section
        padded_sequence = np.zeros((sequence.shape[0] + text_height, sequence.shape[1], 3), dtype=np.uint8)
        padded_sequence[:-text_height] = sequence
        
        # Add analysis text
        y_pos = sequence.shape[0] + padding
        cv2.putText(padded_sequence, f"Analysis: {analysis}",
                   (padding, y_pos), font, font_scale, (255, 255, 255), thickness)
            
        # Save the sequence
        output_path = f"confirmed_fall_{timestamp_str}.jpg"
        cv2.imwrite(output_path, padded_sequence)
        
        # Save the analysis
        analysis_path = f"fall_analysis_{timestamp_str}.txt"
        with open(analysis_path, "w") as f:
            f.write(analysis)
        
        logging.error(f"🚨 LLM CONFIRMED FALL at {timestamp_str}")
        logging.info(f"Annotated sequence saved to: {output_path}")
        logging.info(f"Analysis saved to: {analysis_path}")
        
    except Exception as e:
        logging.error(f"Error processing confirmed fall: {str(e)}")

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