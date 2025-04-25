import asyncio
import logging
import os
import base64
from pathlib import Path

from dotenv import load_dotenv

from elder_care_vision.config.logging_config import setup_logging
from elder_care_vision.services.openai_service import OpenAIService

load_dotenv()

setup_logging()

logger = logging.getLogger(__name__)


async def test_joke(openai_service: OpenAIService) -> None:
    """Test OpenAI service with a joke request."""
    messages = [
        {"role": "system", "content": "You are a helpful and crazy assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
    try:
        response = await openai_service.chat(messages=messages)
        logger.info("Joke test response: %s", response["content"])
    except Exception:
        logger.exception("An error occurred in joke test")


async def test_image_description(openai_service: OpenAIService) -> None:
    """Test OpenAI service with image description."""
    image_path = Path("static/Elderly-Falls.jpg")
    if not image_path.exists():
        logger.error("Test image not found at %s", image_path)
        return

    try:
        # Open and encode the image as base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Prepare the message for OpenAI
        messages = [
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "what's in this image?" },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ]

        # Get the description
        response = await openai_service.chat(messages=messages)
        logger.info("Image description test response: %s", response["content"])

    except Exception:
        logger.exception("An error occurred in image description test")


async def main() -> None:
    """Run all OpenAI service tests."""
    openai_service = OpenAIService()
    
    # Run joke test
    logger.info("Running joke test...")
    await test_joke(openai_service)
    
    # Run image description test
    logger.info("Running image description test...")
    await test_image_description(openai_service)


if __name__ == "__main__":
    asyncio.run(main())
