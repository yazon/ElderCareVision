"""The grand conductor of your fall detection drama! 🎭

This module implements the OopsieController - the mastermind behind your fall detection ecosystem.
It orchestrates the overly-sensitive detector (oopsie_alert) and the sensible filter (oopsie_nanny)
to create a balanced and effective fall detection system.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

from .oopsie_alert.oopsie_alert import FallDetector
from .oopsie_nanny.oopsie_nanny import ImageRecognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class OopsieController:
    """The main controller that orchestrates fall detection and verification.
    
    This class integrates the sensitive fall detector (OopsieAlert) with the rational
    verification system (OopsieNanny) to create a balanced fall detection system.
    
    Attributes:
        alert: The sensitive fall detector that triggers on potential falls
        nanny: The rational verifier that confirms if a fall is real
        warning_frames: Number of frames to show warning when fall is confirmed
        current_warning_frames: Counter for current warning display
        fall_confirmed: Whether the current detection has been confirmed
        is_processing_video: Whether the controller is processing a video
    """
    
    def __init__(self):
        """Initialize the OopsieController with its components."""
        self.alert = FallDetector()
        self.nanny = ImageRecognizer()
        self.warning_frames = 5
        self.current_warning_frames = 0
        self.fall_confirmed = False
        self.is_processing_video = False
        logger.info("OopsieController initialized and ready for fall detection")
        
    def draw_poi(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Draw Point of Interest markers and calculation points on detected persons.
        
        Args:
            frame: The video frame to draw on
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Frame with all calculation points and thresholds drawn
        """
        if landmarks is None:
            return frame
            
        height, width = frame.shape[:2]
        
        # Get all key points used in calculations
        left_shoulder = landmarks.landmark[self.alert.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.alert.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.alert.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.alert.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Convert normalized coordinates to pixel coordinates
        ls_x = int(left_shoulder.x * width)
        ls_y = int(left_shoulder.y * height)
        rs_x = int(right_shoulder.x * width)
        rs_y = int(right_shoulder.y * height)
        lh_x = int(left_hip.x * width)
        lh_y = int(left_hip.y * height)
        rh_x = int(right_hip.x * width)
        rh_y = int(right_hip.y * height)
        
        # Calculate center points
        shoulder_center_x = (ls_x + rs_x) // 2
        shoulder_center_y = (ls_y + rs_y) // 2
        hip_center_x = (lh_x + rh_x) // 2
        hip_center_y = (lh_y + rh_y) // 2
        
        # Draw shoulder points and line
        cv2.circle(frame, (ls_x, ls_y), 5, (0, 255, 255), -1)  # Left shoulder (cyan)
        cv2.circle(frame, (rs_x, rs_y), 5, (0, 255, 255), -1)  # Right shoulder (cyan)
        cv2.line(frame, (ls_x, ls_y), (rs_x, rs_y), (0, 255, 255), 2)  # Shoulder line
        
        # Draw hip points and line
        cv2.circle(frame, (lh_x, lh_y), 5, (255, 255, 0), -1)  # Left hip (yellow)
        cv2.circle(frame, (rh_x, rh_y), 5, (255, 255, 0), -1)  # Right hip (yellow)
        cv2.line(frame, (lh_x, lh_y), (rh_x, rh_y), (255, 255, 0), 2)  # Hip line
        
        # Draw center points
        cv2.circle(frame, (shoulder_center_x, shoulder_center_y), 8, (0, 255, 0), -1)  # Shoulder center (green)
        cv2.circle(frame, (hip_center_x, hip_center_y), 8, (0, 255, 0), -1)  # Hip center (green)
        
        # Draw vertical distance line
        cv2.line(frame, 
                (shoulder_center_x, shoulder_center_y),
                (hip_center_x, hip_center_y),
                (255, 0, 255), 2)  # Vertical distance line (magenta)
        
        # Draw fall threshold line (horizontal line at threshold height)
        threshold_y = int(height * self.alert.fall_threshold)
        cv2.line(frame, 
                (0, threshold_y),
                (width, threshold_y),
                (0, 0, 255), 1)  # Threshold line (red)
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Label shoulder points
        cv2.putText(frame, "LS", (ls_x + 10, ls_y), font, font_scale, (0, 255, 255), thickness)
        cv2.putText(frame, "RS", (rs_x + 10, rs_y), font, font_scale, (0, 255, 255), thickness)
        
        # Label hip points
        cv2.putText(frame, "LH", (lh_x + 10, lh_y), font, font_scale, (255, 255, 0), thickness)
        cv2.putText(frame, "RH", (rh_x + 10, rh_y), font, font_scale, (255, 255, 0), thickness)
        
        # Label centers
        cv2.putText(frame, "SC", (shoulder_center_x + 10, shoulder_center_y), font, font_scale, (0, 255, 0), thickness)
        cv2.putText(frame, "HC", (hip_center_x + 10, hip_center_y), font, font_scale, (0, 255, 0), thickness)
        
        # Label threshold
        cv2.putText(frame, f"Fall Threshold: {self.alert.fall_threshold:.2f}", 
                   (10, threshold_y - 10), font, font_scale, (0, 0, 255), thickness)
        
        return frame
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Process a single frame for fall detection and verification.
        
        Args:
            frame: The video frame to process
            
        Returns:
            Tuple containing:
            - The processed frame with any warnings
            - Boolean indicating if a fall was confirmed
        """
        # Convert frame to RGB for pose detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe pose detection
        results = self.alert.pose.process(rgb_frame)
        
        # Draw POI if person is detected
        if results.pose_landmarks:
            frame = self.draw_poi(frame, results.pose_landmarks)
            
        # Check for potential fall
        if results.pose_landmarks:
            potential_fall = self.alert.detect_fall(results.pose_landmarks)
            
            if potential_fall and not self.fall_confirmed:
                logger.info("Potential fall detected!")
                if not self.is_processing_video:
                    # For single images, use nanny verification
                    temp_path = "temp_fall_frame.jpg"
                    cv2.imwrite(temp_path, frame)
                    
                    try:
                        # Verify with nanny
                        analysis = self.nanny.analyze_image(temp_path)
                        if "High" in analysis:
                            logger.info("Fall confirmed by nanny verification")
                            self.fall_confirmed = True
                            self.current_warning_frames = self.warning_frames
                        else:
                            logger.info("Fall not confirmed by nanny verification")
                    except Exception as e:
                        logger.error(f"Error verifying fall: {str(e)}")
                    finally:
                        # Clean up temp file
                        Path(temp_path).unlink(missing_ok=True)
                else:
                    # For video, trust the alert system
                    logger.info("Fall confirmed in video processing")
                    self.fall_confirmed = True
                    self.current_warning_frames = self.warning_frames
        
        # Add warning overlay if fall is confirmed
        if self.fall_confirmed and self.current_warning_frames > 0:
            frame = self.alert.add_warning_overlay(frame)
            self.current_warning_frames -= 1
            if self.current_warning_frames == 0:
                logger.info("Fall warning period ended")
                self.fall_confirmed = False
                
        return frame, self.fall_confirmed
        
    def process_video(self, video_path: str) -> None:
        """Process a video file for fall detection.
        
        Args:
            video_path: Path to the video file to process
        """
        logger.info(f"Starting video processing: {video_path}")
        self.is_processing_video = True
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return
            
        try:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames
                    logger.debug(f"Processing frame {frame_count}")
                    
                # Process frame
                processed_frame, fall_detected = self.process_frame(frame)
                
                # Display result
                cv2.imshow("Fall Detection", processed_frame)
                
                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Video processing stopped by user")
                    break
        finally:
            self.is_processing_video = False
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Video processing completed")
        
    def process_image(self, image_path: str) -> Tuple[np.ndarray, bool]:
        """Process a single image for fall detection.
        
        Args:
            image_path: Path to the image file to process
            
        Returns:
            Tuple containing:
            - The processed image with any warnings
            - Boolean indicating if a fall was detected
        """
        self.is_processing_video = False
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        return self.process_frame(frame)
