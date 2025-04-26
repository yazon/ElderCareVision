"""
Fall Detection System using MediaPipe

This system detects falls in videos by analyzing human pose landmarks using MediaPipe's
pose estimation technology. It provides real-time visual feedback when falls are detected.

Key Features:
- Real-time pose estimation and tracking
- Multi-criteria fall detection
- Visual warning system
- Configurable detection parameters

Author: Your Name
Date: Current Date
"""

import cv2
import numpy as np
import sys
import mediapipe as mp


class FallDetector:
    """
    A class for detecting falls in video using MediaPipe pose estimation.

    This class implements a fall detection system that:
    1. Processes video frames using MediaPipe's pose estimation
    2. Analyzes body position and orientation
    3. Detects falls based on multiple criteria
    4. Provides visual feedback when falls are detected

    Attributes:
        mp_pose: MediaPipe pose estimation model
        pose: MediaPipe pose detection instance
        fall_threshold: Threshold for vertical position to detect fall
        min_confidence: Minimum confidence for keypoint detection
        min_keypoints_visible: Minimum number of keypoints that must be visible
        frame_skip: Number of frames to skip between detections
        target_width: Target width for frame resizing
    """

    def __init__(self):
        """Initialize the FallDetector with optimized parameters."""
        # Initialize MediaPipe Pose with optimized settings
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Process video frames
            model_complexity=1,  # Use medium complexity for better speed
            enable_segmentation=False,  # Disable segmentation for speed
            min_detection_confidence=0.7,  # Increased from 0.5 to 0.7 for better accuracy
            min_tracking_confidence=0.7,  # Increased from 0.5 to 0.7 for better tracking
        )

        # Fall detection parameters - adjusted for less sensitivity
        self.fall_threshold = 0.95  # Increased from 0.85 to 0.95 (must be more horizontal)
        self.min_confidence = 0.7  # Increased from 0.5 to 0.7 for better accuracy
        self.min_keypoints_visible = 8  # Increased from 5 to 8 for more reliable detection
        self.min_duration = 5  # Number of consecutive frames to confirm fall

        # Performance optimization parameters
        self.frame_skip = 2  # Process every nth frame
        self.target_width = 640  # Target width for resizing
        self.frame_counter = 0  # Counter for frame skipping
        self.fall_frames = 0  # Counter for consecutive fall frames

    def resize_frame(self, frame):
        """Resize frame while maintaining aspect ratio for better performance.

        Args:
            frame: Input frame

        Returns:
            Resized frame
        """
        height, width = frame.shape[:2]
        if width > self.target_width:
            scale = self.target_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        return frame

    def add_warning_overlay(self, frame):
        """Add a visual warning overlay to the frame."""
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Add red tint with reduced opacity
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)

        # Add warning text
        font = cv2.FONT_HERSHEY_DUPLEX
        text = "FALL DETECTED!"
        font_scale = min(width / 500, 2.0)  # Scale text based on frame width
        thickness = max(1, int(font_scale * 2))

        # Calculate text size and position
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        # Add text with shadow
        cv2.putText(overlay, text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Blend with reduced opacity
        alpha = 0.3
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def detect_fall(self, landmarks):
        """Detect if a person has fallen based on pose landmarks."""
        if landmarks is None:
            return False

        # Quick check for minimum visible keypoints
        visible_keypoints = sum(1 for landmark in landmarks.landmark if landmark.visibility > self.min_confidence)
        if visible_keypoints < self.min_keypoints_visible:
            return False

        # Get key landmarks
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]

        # Quick visibility check
        if any(l.visibility < self.min_confidence for l in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return False

        # Calculate vertical distance (optimized)
        shoulder_y = (left_shoulder.y + right_shoulder.y) * 0.5
        hip_y = (left_hip.y + right_hip.y) * 0.5
        vertical_distance = abs(shoulder_y - hip_y)

        # Calculate alignment (optimized)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        hip_width = abs(left_hip.x - right_hip.x)
        width_ratio = min(shoulder_width, hip_width) / max(shoulder_width, hip_width)

        # Check if person is actually lying down (not just bending)
        is_horizontal = vertical_distance < self.fall_threshold
        is_aligned = width_ratio > 0.7

        # Check for sudden movement (optional)
        # sudden_movement = self._check_sudden_movement(landmarks)

        # Update fall frame counter
        if is_horizontal and is_aligned:
            self.fall_frames += 1
        else:
            self.fall_frames = 0

        # Only confirm fall after minimum duration
        return self.fall_frames >= self.min_duration

    def process_frame(self, frame):
        """Process a single frame for fall detection.

        Args:
            frame: Input video frame

        Returns:
            Tuple of (processed_frame, fall_detected)
        """
        # Skip frames for performance
        self.frame_counter = (self.frame_counter + 1) % (self.frame_skip + 1)
        if self.frame_counter != 0:
            return frame, False

        # Resize frame for better performance
        frame = self.resize_frame(frame)

        # Convert to RGB (required by MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.pose.process(rgb_frame)

        # Detect fall
        fall_detected = False
        if results.pose_landmarks:
            fall_detected = self.detect_fall(results.pose_landmarks)

            # Draw landmarks (optional, can be disabled for better performance)
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1
                ),
            )

        if fall_detected:
            frame = self.add_warning_overlay(frame)

        return frame, fall_detected

    def process_video(self, video_path):
        """Process a video file for fall detection."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            frame, fall_detected = self.process_frame(frame)

            # Display frame
            cv2.imshow("Fall Detection", frame)

            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()


def main():
    """Main function to run the fall detection system."""
    if len(sys.argv) != 2:
        print("Usage: python fall_detection.py <video_path>")
        return

    video_path = sys.argv[1]
    detector = FallDetector()
    detector.process_video(video_path)


if __name__ == "__main__":
    main()
