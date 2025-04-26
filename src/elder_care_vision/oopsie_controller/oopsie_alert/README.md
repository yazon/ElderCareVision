# Fall Detection System using MediaPipe

This application detects falls in videos using MediaPipe's pose estimation technology. It analyzes human pose landmarks to identify when a person has fallen and provides visual warnings.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How it Works](#how-it-works)
- [Technical Details](#technical-details)
- [Parameters and Tuning](#parameters-and-tuning)
- [Troubleshooting](#troubleshooting)

## Overview

The system uses computer vision and pose estimation to:

1. Track human poses in real-time video
1. Analyze body position and orientation
1. Detect falls based on multiple criteria
1. Provide visual warnings when falls are detected

## System Requirements

- Python 3.6 or higher
- OpenCV (cv2)
- MediaPipe
- NumPy

## Installation

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application with a video file as input:

```bash
python fall_detection.py path/to/your/video.mp4
```

### Controls

- Press 'q' to quit the application
- The application will display "FALL DETECTED!" with a red overlay when a fall is detected

## How it Works

The system uses a multi-stage approach to detect falls:

1. **Pose Estimation**: MediaPipe's pose estimation model tracks 33 keypoints on the human body
1. **Keypoint Validation**: Ensures enough keypoints are visible and have sufficient confidence
1. **Fall Detection Logic**:
   - Analyzes vertical distance between shoulders and hips
   - Checks body alignment to distinguish falls from bending
   - Uses multiple criteria to reduce false positives
1. **Visual Feedback**: Provides immediate visual warning when a fall is detected

## Technical Details

### Fall Detection Algorithm

The system detects falls by analyzing:

1. **Vertical Alignment**: Distance between shoulder and hip landmarks
1. **Horizontal Alignment**: Ratio between shoulder and hip widths
1. **Keypoint Visibility**: Number of visible and confident keypoints
1. **Landmark Confidence**: Individual confidence scores for critical landmarks

### Parameters

Key parameters that can be adjusted:

- `fall_threshold`: Vertical distance threshold (default: 0.85)
- `min_confidence`: Minimum confidence for keypoint detection (default: 0.5)
- `min_keypoints_visible`: Minimum number of visible keypoints (default: 5)
- `width_ratio_threshold`: Minimum ratio for shoulder-hip alignment (default: 0.7)

## Parameters and Tuning

### Adjusting Sensitivity

To adjust the system's sensitivity:

1. **Reduce False Positives**:

   - Increase `fall_threshold` (e.g., 0.9)
   - Increase `min_confidence` (e.g., 0.6)
   - Increase `min_keypoints_visible` (e.g., 7)

1. **Increase Sensitivity**:

   - Decrease `fall_threshold` (e.g., 0.8)
   - Decrease `min_confidence` (e.g., 0.4)
   - Decrease `min_keypoints_visible` (e.g., 3)

### Visual Feedback

The warning overlay:

- Appears for 5 frames (approximately 0.17 seconds at 30fps)
- Uses a semi-transparent red background (40% opacity)
- Displays two warning messages with shadow effects
- Includes pose landmarks for visual verification

## Troubleshooting

Common issues and solutions:

1. **No Detection**:

   - Ensure good lighting conditions
   - Check if the person is fully visible in frame
   - Verify camera angle is appropriate

1. **False Positives**:

   - Adjust detection parameters as described above
   - Ensure the camera is stable and not moving

1. **Performance Issues**:

   - Reduce video resolution if needed
   - Close other resource-intensive applications

## Future Improvements

Potential enhancements:

1. Add audio alerts
1. Implement fall severity classification
1. Add support for multiple people
1. Integrate with emergency response systems
1. Add fall prevention analysis
