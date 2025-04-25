import asyncio
import logging

from dotenv import load_dotenv

from elder_care_vision.config.logging_config import setup_logging
from elder_care_vision.core.coordinator import CoordinatorAgent

load_dotenv()

setup_logging()

logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info("Starting Elder Care Vision system...")
    coordinator = CoordinatorAgent()
    await coordinator.run()


if __name__ == "__main__":
    logger.info("Starting Elder Care Vision system...")
    asyncio.run(main())
