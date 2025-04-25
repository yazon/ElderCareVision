#!/usr/bin/env python3

import cv2
import logging
from pathlib import Path
from datetime import datetime
from .oopsie_controller import OopsieController

def manual_camera_test(
    camera_id: int = 0,
    output_dir: str = "test_output"
) -> None:
    """
    Run a manual test of the OopsieController with live camera feed.
    
    Controls:
    - Q: Quit
    - S: Save current frame
    - D: Toggle debug overlay
    
    Args:
        camera_id: ID of the camera to use (default: 0)
        output_dir: Directory to save test outputs (default: "test_output")
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize camera
    logger.info(f"Opening camera {camera_id}")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_id}")
    
    # Initialize controller
    controller = OopsieController()
    
    # Add example subscribers for testing
    def on_algorithm_fall(frame, timestamp):
        logger.info(f"Algorithm detected fall at {timestamp}")
        save_path = output_path / f"algorithm_fall_{timestamp}.jpg"
        cv2.imwrite(str(save_path), frame)
        logger.info(f"Saved fall detection frame to {save_path}")
        
    def on_confirmed_fall(frame_sequence, analysis):
        logger.info(f"LLM confirmed fall: {analysis}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = output_path / f"confirmed_fall_{timestamp}.jpg"
        cv2.imwrite(str(save_path), frame_sequence[-1])  # Save last frame
        logger.info(f"Saved confirmed fall frame to {save_path}")
    
    controller.add_algorithm_fall_subscriber(on_algorithm_fall)
    controller.add_confirmed_fall_subscriber(on_confirmed_fall)
    
    debug_overlay = True
    frame_count = 0
    logger.info("Starting manual test. Press 'Q' to quit, 'S' to save frame, 'D' to toggle debug overlay")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Process frame
            processed_frame, fall_detected = controller.process_frame(frame)
            frame_count += 1
            
            # Show debug info
            if debug_overlay:
                cv2.putText(
                    processed_frame,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
                if fall_detected:
                    cv2.putText(
                        processed_frame,
                        "FALL DETECTED!",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
            
            # Display frame
            cv2.imshow("ElderCareVision Test", processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quitting...")
                break
            elif key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = output_path / f"test_frame_{timestamp}.jpg"
                cv2.imwrite(str(save_path), processed_frame)
                logger.info(f"Saved frame to {save_path}")
            elif key == ord("d"):
                debug_overlay = not debug_overlay
                logger.info(f"Debug overlay: {'on' if debug_overlay else 'off'}")
                
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Test completed")

if __name__ == "__main__":
    manual_camera_test() 