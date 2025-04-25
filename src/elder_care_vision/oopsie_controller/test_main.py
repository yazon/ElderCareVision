import cv2
import pytest
import time
import logging
from pathlib import Path
from unittest.mock import MagicMock
from .main import process_video
from .oopsie_controller import OopsieController

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestOopsieController:
    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create temporary directory for test outputs"""
        return tmp_path / "test_output"

    @pytest.fixture
    def mock_subscribers(self):
        """Create mock subscribers for testing"""
        class Subscribers:
            def __init__(self):
                self.algorithm_falls = []
                self.confirmed_falls = []
                
            def on_algorithm_fall(self, frame, timestamp):
                self.algorithm_falls.append((frame, timestamp))
                logger.info(f"Algorithm fall detected at {timestamp}")
                
            def on_confirmed_fall(self, frame_sequence, analysis):
                self.confirmed_falls.append((frame_sequence, analysis))
                logger.info(f"Confirmed fall with analysis: {analysis}")
        
        return Subscribers()

    def test_camera_initialization(self):
        """Test if camera can be initialized"""
        cap = cv2.VideoCapture(0)
        try:
            assert cap.isOpened(), "Failed to open camera"
            ret, frame = cap.read()
            assert ret, "Failed to read frame from camera"
            assert frame is not None, "Frame is None"
            assert frame.shape[0] > 0 and frame.shape[1] > 0, "Invalid frame dimensions"
        finally:
            cap.release()

    def test_frame_processing(self, output_dir, mock_subscribers):
        """Test frame processing with real camera"""
        output_dir.mkdir(exist_ok=True)
        
        # Initialize controller
        controller = OopsieController(
            output_dir=str(output_dir),
            display=True
        )
        
        # Add subscribers
        controller.add_algorithm_fall_subscriber(mock_subscribers.on_algorithm_fall)
        controller.add_confirmed_fall_subscriber(mock_subscribers.on_confirmed_fall)
        
        # Open camera
        cap = cv2.VideoCapture(0)
        try:
            assert cap.isOpened(), "Failed to open camera"
            
            # Process frames for 10 seconds
            start_time = time.time()
            frames_processed = 0
            
            while time.time() - start_time < 10:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                controller.process_frame(frame)
                frames_processed += 1
                
                # Display frame with overlay
                cv2.imshow('Test Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            logger.info(f"Processed {frames_processed} frames in 10 seconds")
            assert frames_processed > 0, "No frames were processed"
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            controller.cleanup()

    def test_queue_performance(self, output_dir):
        """Test queue performance with real camera"""
        controller = OopsieController(
            output_dir=str(output_dir),
            display=True
        )
        
        cap = cv2.VideoCapture(0)
        try:
            assert cap.isOpened(), "Failed to open camera"
            
            # Process frames for 5 seconds and monitor queue sizes
            start_time = time.time()
            max_queue_size = 0
            
            while time.time() - start_time < 5:
                ret, frame = cap.read()
                if not ret:
                    break
                
                controller.process_frame(frame)
                current_queue_size = controller.get_queue_size()
                max_queue_size = max(max_queue_size, current_queue_size)
                
                # Display frame
                cv2.imshow('Queue Test', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            logger.info(f"Maximum queue size: {max_queue_size}")
            assert max_queue_size < 30, "Queue size grew too large"
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            controller.cleanup()

def test_live_camera():
    """Integration test for live camera processing"""
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        process_video(
            video_source=0,  # Use default camera
            output_dir=str(output_dir),
            display=True,
            record=False,
            debug=True
        )
    except KeyboardInterrupt:
        logger.info("Test stopped by user")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the live camera test directly
    logging.basicConfig(level=logging.INFO)
    test_live_camera() 