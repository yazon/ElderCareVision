"""Module for analyzing images of elderly falls and generating appropriate notifications.

This module provides functionality to analyze images of elderly falls using OpenAI's
vision capabilities. It processes images to detect falls and generates appropriate
notification messages for family members or caregivers.

Example:
    ```python
    from elder_care_vision.core.tools.emergency_call.img_analizer import ImgAnalizer

    # Initialize the analyzer
    analyzer = ImgAnalizer()

    # Analyze an image
    notification = await analyzer.analize_image(base64_image_string)
    print(notification)
    ```
"""

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
    """Class for analyzing images of elderly falls and generating notifications.

    This class uses OpenAI's vision capabilities to analyze images of elderly falls
    and generate appropriate notification messages for family members or caregivers.

    Attributes:
        openai_service (OpenAIService): Instance of OpenAIService for image analysis.

    Example:
        ```python
        analyzer = ImgAnalizer()
        notification = await analyzer.analize_image(base64_image_string)
        print(notification)
        ```
    """

    def __init__(self) -> None:
        """Initialize the ImgAnalizer with an OpenAIService instance."""
        self.openai_service = OpenAIService()

    async def analize_image(self, base64_image: str) -> str:
        """Analyze an image of an elderly fall and generate a notification message.

        This method takes a base64-encoded image, sends it to OpenAI for analysis,
        and generates an appropriate notification message for family members or
        caregivers.

        Args:
            base64_image (str): Base64-encoded string of the image to analyze.

        Returns:
            str: The generated notification message.

        Example:
            ```python
            with open("fall_image.jpg", "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                notification = await analyzer.analize_image(base64_image)
                print(notification)
            ```

        Note:
            The method logs the generated notification message using the logger.
            Any errors during analysis are caught and logged as exceptions.
        """
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
            return response["content"]

        except Exception:
            logger.exception("An error occurred during image analysis")
            return ""


if __name__ == "__main__":
    image_path = Path("static/Elderly-Falls.jpg")

    if not image_path.exists():
        logger.error("Test image not found at %s", image_path)
        exit(0)

    base64_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    asyncio.run(ImgAnalizer().analize_image(base64_image))
