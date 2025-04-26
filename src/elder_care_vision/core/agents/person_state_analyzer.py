"""Placeholder for the Person State Analyzer agent."""

import cv2
import logging
from datetime import datetime, UTC
from pathlib import Path
import threading
import asyncio

import numpy as np

from agents import function_tool
from pydantic import BaseModel

from elder_care_vision.oopsie_controller.oopsie_controller import OopsieController

logger = logging.getLogger(__name__)


@function_tool
def analyze_image(image_data: str) -> str:
    """Analyze the image and return a string describing the person's state."""
    logger.info(f"Analyzing image: {image_data}")
    return "Person is laying on the floor"


class PersonState(BaseModel):
    fall_confidence: int
    """Represents the confidence of a fall."""


class PersonStateAnalyzerAgent:
    """Analyzes the state of a person based on camera data."""

    def __init__(self, camera_id: int = 0) -> None:
        """Initializes the Person State Analyzer agent."""
        logger.info("Initializing Person State Analyzer Agent")
        self.confidence_level = 0
        self.camera_id = camera_id

        # Create output directory
        self.output_path = Path("test_output")
        self.output_path.mkdir(exist_ok=True)

        # Initialize controller
        self.controller = OopsieController()

        # Add example subscribers for testing
        def on_algorithm_fall(frame, timestamp) -> None:
            logger.info(f"Algorithm detected fall at {timestamp}")
            save_path = self.output_path / f"algorithm_fall_{timestamp}.jpg"
            cv2.imwrite(str(save_path), frame)
            logger.info(f"Saved fall detection frame to {save_path}")

        def on_confirmed_fall(frame_sequence: list[np.ndarray], analysis: str) -> None:
            logger.info(f"LLM confirmed fall: {analysis}")
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            save_path = self.output_path / f"confirmed_fall_{timestamp}.jpg"
            cv2.imwrite(str(save_path), frame_sequence[-1])  # Save last frame
            logger.info(f"Saved confirmed fall frame to {save_path}")

        self.controller.add_algorithm_fall_subscriber(on_algorithm_fall)
        self.controller.add_confirmed_fall_subscriber(on_confirmed_fall)

    async def process_image(self) -> None:
        """Run the Person State Analyzer agent."""
        debug_overlay = True
        frame_count = 0

        # Initialize camera
        logger.info(f"Opening camera {self.camera_id}")
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            msg = f"Failed to open camera {self.camera_id}"
            logger.error(msg)
            raise RuntimeError(msg)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                # Process frame
                processed_frame, confidence_level = self.controller.process_frame(frame)
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
                        2,
                    )
                    if confidence_level >= 70:
                        cv2.putText(
                            processed_frame, "FALL DETECTED!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                        )

                # Display frame
                cv2.imshow("ElderCareVision Test", processed_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quitting...")
                    break
                if key == ord("s"):
                    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                    save_path = self.output_path / f"test_frame_{timestamp}.jpg"
                    cv2.imwrite(str(save_path), processed_frame)
                    logger.info(f"Saved frame to {save_path}")
                elif key == ord("d"):
                    debug_overlay = not debug_overlay
                    logger.info(f"Debug overlay: {"on" if debug_overlay else "off"}")

        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        except Exception as e:
            logger.error(f"Test failed: {e!s}")
            raise
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    async def run(self) -> int:
        """Run the Person State Analyzer agent in a separate thread."""

        def run_process_image() -> None:
            """Run process_image in a new event loop."""
            asyncio.run(self.process_image())

        thread = threading.Thread(target=run_process_image)
        thread.daemon = True  # Allow program to exit even if thread is running
        thread.start()

        return 0
