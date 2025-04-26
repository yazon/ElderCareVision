import argparse
import asyncio
import logging

from dotenv import load_dotenv

from elder_care_vision.config.logging_config import setup_logging
from elder_care_vision.core.coordinator import Coordinator

load_dotenv()

setup_logging()

logger = logging.getLogger(__name__)


async def main(video_source: int | str = 0) -> None:
    """
    Main entry point for the Elder Care Vision system.

    Args:
        video_source: Either a camera ID (integer) or an RTSP stream URL (string).
                     Defaults to 0 (default camera).
    """
    logger.info("Starting Elder Care Vision system...")
    coordinator = Coordinator(video_source)
    await coordinator.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elder Care Vision System")
    parser.add_argument(
        "--video-source",
        type=str,
        default="0",
        help="Video source (camera ID or RTSP URL). Default: 0 (default camera)",
    )
    args = parser.parse_args()

    # Convert video source to int if it's a number, otherwise keep as string
    try:
        video_source = int(args.video_source)
    except ValueError:
        video_source = args.video_source

    logger.info("Starting Elder Care Vision system...")
    asyncio.run(main(video_source))
