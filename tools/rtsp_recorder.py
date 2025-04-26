#!/usr/bin/env python3

import cv2
import argparse
import time
import logging
from datetime import datetime
from urllib.parse import quote

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("rtsp_recorder.log")],
)


def record_rtsp_stream(
    rtsp_url: str,
    duration_seconds: int,
    output_file: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> None:
    """
    Record video from RTSP stream for specified duration.

    Args:
        rtsp_url: RTSP stream URL
        duration_seconds: Duration to record in seconds
        output_file: Output file path. If None, generates timestamp-based filename
        username: Username for RTSP authentication (optional)
        password: Password for RTSP authentication (optional)
    """
    # Construct authenticated URL if credentials are provided
    if username:
        # Parse the URL and add credentials
        if "rtsp://" in rtsp_url:
            base_url = rtsp_url.replace("rtsp://", "")
            if username and password:
                rtsp_url = f"rtsp://{quote(username)}:{quote(password)}@{base_url}"
            elif username:
                rtsp_url = f"rtsp://{quote(username)}:@{base_url}"

    logging.info(f"Connecting to RTSP stream: {rtsp_url}")

    # Set OpenCV to use TCP transport for more reliable streaming
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout

    if not cap.isOpened():
        logging.error(f"Error: Could not open RTSP stream: {rtsp_url}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    logging.info(f"Stream properties - Width: {width}, Height: {height}, FPS: {fps}")

    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"recording_{timestamp}.mp4"

    # Create video writer with better codec options
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    logging.info(f"Recording started. Duration: {duration_seconds} seconds")
    logging.info(f"Output file: {output_file}")

    start_time = time.time()
    frame_count = 0
    error_count = 0
    max_errors = 10  # Maximum number of consecutive errors before stopping

    try:
        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()

            if not ret:
                error_count += 1
                logging.warning(f"Error reading frame. Error count: {error_count}")
                if error_count >= max_errors:
                    logging.error("Too many consecutive errors. Stopping recording.")
                    break
                time.sleep(0.1)  # Small delay before retrying
                continue

            # Reset error count on successful frame read
            error_count = 0
            frame_count += 1

            out.write(frame)

            # Display recording progress
            elapsed = time.time() - start_time
            print(f"\rRecording progress: {elapsed:.1f}/{duration_seconds} seconds", end="")

    except KeyboardInterrupt:
        logging.info("Recording stopped by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    finally:
        cap.release()
        out.release()
        logging.info(f"Recording completed. Total frames recorded: {frame_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record video from RTSP stream")
    parser.add_argument("rtsp_url", help="RTSP stream URL")
    parser.add_argument("duration", type=int, help="Recording duration in seconds")
    parser.add_argument("--output", "-o", help="Output file path (optional)")
    parser.add_argument("--username", "-u", help="Username for RTSP authentication (optional)")
    parser.add_argument("--password", "-p", help="Password for RTSP authentication (optional)")

    args = parser.parse_args()

    record_rtsp_stream(args.rtsp_url, args.duration, args.output, args.username, args.password)


if __name__ == "__main__":
    main()
