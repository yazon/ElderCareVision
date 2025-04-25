# ElderCareVision

ElderCareVision is an advanced fall detection system that uses computer vision and AI to monitor and detect falls in real-time. The system combines pose detection, machine learning, and LLM-based analysis to provide accurate fall detection with minimal false positives.

## Features

- **Real-time Fall Detection**: Continuous monitoring using advanced pose detection
- **AI-Powered Analysis**: Two-stage verification with algorithm detection and LLM confirmation
- **Non-blocking Video Processing**: Queue-based system for smooth video playback
- **Flexible Alert System**: Customizable subscriber system for fall notifications
- **Visual Feedback**: On-screen display of detection status and system metrics
- **Frame History**: Automatic capture and storage of fall events
- **Performance Monitoring**: Real-time queue statistics and system status

## Quick Start

1. Install the package:
```bash
pip install -e .
```

2. Run with a video file:
```bash
python -m elder_care_vision.oopsie_controller.main --video path/to/video.mp4
```

3. Run with webcam:
```bash
python -m elder_care_vision.oopsie_controller.main --camera 0
```

## Configuration Options

The OopsieController supports several command-line arguments:

```bash
--video PATH          Path to input video file
--camera ID          Camera device ID (default: 0)
--display            Enable visual display (default: True)
--record PATH        Record output video to specified path
--debug             Enable debug logging
--cooldown SECONDS   LLM analysis cooldown period (default: 10)
```

## System Components

### OopsieController
The main controller that orchestrates:
- Frame acquisition and processing
- Pose detection and analysis
- Fall detection logic
- LLM-based verification
- Subscriber notifications

### Frame Queue System
- Non-blocking frame processing
- Real-time performance monitoring
- Automatic queue size management

### Fall Detection
Two-stage verification process:
1. Algorithm-based detection using pose analysis
2. LLM-based confirmation using frame sequences

### Subscriber System
Subscribe to two types of events:
1. Algorithm-detected falls:
```python
controller.add_algorithm_fall_subscriber(your_callback)
```

2. LLM-confirmed falls:
```python
controller.add_confirmed_fall_subscriber(your_callback)
```

Example subscriber implementation:
```python
def on_fall_detected(frame, timestamp):
    # Handle fall detection
    cv2.imwrite(f"fall_{timestamp}.jpg", frame)
    logging.error(f"Fall detected at {timestamp}")
```

## Output Files

The system generates the following files when falls are detected:

- `algorithm_fall_YYYYMMDD-HHMMSS.jpg`: Frames from algorithm-detected falls
- `confirmed_fall_YYYYMMDD-HHMMSS.jpg`: Frames from LLM-confirmed falls with analysis text

## Performance Optimization

- Use GPU acceleration when available
- Adjust queue sizes based on system capabilities
- Monitor queue statistics in debug mode
- Set appropriate LLM cooldown periods

## Troubleshooting

Common issues and solutions:

1. Video playback is laggy:
   - Check queue statistics
   - Reduce frame resolution
   - Increase process thread count

2. False positives:
   - Adjust pose detection confidence threshold
   - Increase LLM cooldown period
   - Fine-tune fall detection parameters

3. Missing fall events:
   - Check frame queue size
   - Verify subscriber connections
   - Enable debug logging

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) for details.

## Architecture

For detailed system architecture information, please refer to:
- [Architecture Documentation](docs/architecture.md)
- [Sequence Diagram](docs/sequence_diagram.puml)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MediaPipe for pose detection
- OpenCV for video processing
- OpenAI for LLM capabilities
