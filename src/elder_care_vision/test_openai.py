import asyncio
import logging

from dotenv import load_dotenv

from elder_care_vision.config.logging_config import setup_logging
from elder_care_vision.services.openai_service import OpenAIService

load_dotenv()

setup_logging()

logger = logging.getLogger(__name__)


async def main() -> None:
    """Example usage of OpenAIService."""
    openai_service = OpenAIService()
    messages = [
        {"role": "system", "content": "You are a helpful and crazy assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
    try:
        response = await openai_service.chat(messages=messages)
        logger.info("OpenAI response: %s", response["content"])
    except Exception:
        logger.exception("An error occurred")


if __name__ == "__main__":
    asyncio.run(main())
