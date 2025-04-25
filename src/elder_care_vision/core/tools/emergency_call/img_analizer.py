import asyncio
import base64
import logging
from pathlib import Path

from dotenv import load_dotenv

from elder_care_vision.config.logging_config import setup_logging
from elder_care_vision.services.openai_service import OpenAIService

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


class ImgAnalizer:
    def __init__(self) -> None:
        self.openai_service = OpenAIService()

    async def analize_image(self, base64_image: str) -> None:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Based on the attached image showing an elderly person who has fallen and is not "
                                "responding to voice commands, generate a short, empathetic, and informative notification "
                                "message intended for a grandson. The message should briefly describe the incident, indicate "
                                "that assistance may be needed, and ask the grandson to check on the elderly person."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ]

            response = await self.openai_service.chat(messages=messages)
            logger.info("Image description response: %s", response["content"])

        except Exception:
            logger.exception("An error occurred during image analysis")


if __name__ == "__main__":
    image_path = Path("static/Elderly-Falls.jpg")

    if not image_path.exists():
        logger.error("Test image not found at %s", image_path)
        exit(0)

    base64_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    asyncio.run(ImgAnalizer().analize_image(base64_image))
