"""Placeholder for the Person State Analyzer agent."""

import asyncio
import base64
import logging
import threading
from datetime import UTC, datetime

import cv2
import numpy as np

from elder_care_vision.core.agents.psa_data import FallDetectionResult
from elder_care_vision.oopsie_controller.oopsie_controller import OopsieController
from elder_care_vision.utils.utils import get_static_path

logger = logging.getLogger(__name__)


class PersonStateAnalyzerAgent:
    """
    Analyzes person state using computer vision and fall detection algorithms.

    Handles camera input processing, fall detection, and event notifications.
    Manages subscribers for different types of fall detection events.

    Attributes:
        camera_id: ID of the video capture device
        fall_detection_result: Current fall detection status and metadata
        output_path: Directory for saving detection outputs
        controller: Main processing controller instance
    """

    def __init__(self, camera_id: int = 0) -> None:
        """
        Initialize analyzer agent with camera and event handlers.

        Args:
            camera_id: Numeric identifier for video capture device. Defaults to 0.

        Raises:
            RuntimeError: If camera initialization fails
        """
        logger.info("Initializing Person State Analyzer Agent")
        self.camera_id = camera_id
        self.fall_detection_result = FallDetectionResult()
        # Create output directory
        self.output_path = get_static_path()
        self.output_path.mkdir(exist_ok=True)

        # Initialize controller
        self.controller = OopsieController()

        # Add example subscribers for testing
        def on_algorithm_fall(frame: any, _: any, timestamp: any) -> None:
            """
            Callback for initial algorithm-based fall detection.

            Args:
                frame: Captured video frame with detected fall
                _: Unused confidence parameter
                timestamp: Detection timestamp
            """
            logger.info(f"Algorithm detected fall at {timestamp}")
            save_path = self.output_path / f"algorithm_fall_{timestamp}.jpg"
            cv2.imwrite(str(save_path), frame)
            logger.info(f"Saved fall detection frame to {save_path}")

        def on_confirmed_fall(
            frame_sequence: list[np.ndarray], _: list[float], analysis: str, confidence_level: int
        ) -> None:
            """
            Callback for confirmed fall events after LLM validation.

            Args:
                frame_sequence: List of video frames around detection time
                _: Unused confidence scores list
                analysis: Textual description of fall analysis
                confidence_level: Final confidence score (0-100)
            """
            logger.info(f"LLM confirmed fall {analysis} with confidence level {confidence_level}")
            self.fall_detection_result.confidence_level = confidence_level
            self.fall_detection_result.analysis = analysis
            # Encode last frame
            _, buffer = cv2.imencode(".jpg", frame_sequence[-1])
            self.fall_detection_result.fall_image = base64.b64encode(buffer).decode("utf-8")
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            save_path = self.output_path / f"confirmed_fall_{timestamp}.jpg"
            cv2.imwrite(str(save_path), frame_sequence[-1])  # Save last frame
            logger.info(f"Saved confirmed fall frame to {save_path}")

        self.controller.add_algorithm_fall_subscriber(on_algorithm_fall)
        self.controller.add_confirmed_fall_subscriber(on_confirmed_fall)

    async def process_image(self) -> None:
        """
        Main processing loop for camera frames.

        Handles:
        - Camera initialization and frame capture
        - Frame processing through controller
        - Debug overlay rendering
        - System shutdown cleanup

        Raises:
            RuntimeError: If frame processing fails unexpectedly
        """
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
                _ = cv2.waitKey(1) & 0xFF

        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        except Exception as e:
            msg = f"Test failed: {e!s}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    async def run(self) -> int:
        """
        Start the analyzer in a background thread.

        Returns:
            int: Always returns 0 to indicate successful thread start

        Note:
            Runs as daemon thread to allow clean program exit
        """

        def run_process_image() -> None:
            """Helper function to run async process in new event loop."""
            asyncio.run(self.process_image())

        thread = threading.Thread(target=run_process_image)
        thread.daemon = True  # Allow program to exit even if thread is running
        thread.start()

        return 0
