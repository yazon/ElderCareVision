"""Example usage of ADBPhoneCallManager for making and managing phone calls."""

import base64
import logging
import asyncio
from pathlib import Path

from elder_care_vision.config.logging_config import setup_logging
from elder_care_vision.core.tools.emergency_call.adb_phone_call_manager import ADBPhoneCallManager
from elder_care_vision.core.tools.emergency_call.img_analizer import ImgAnalizer
from elder_care_vision.services.openai_service import OpenAIService
# from elder_care_vision.core.tools.function_tool import function_tool

setup_logging()

logger = logging.getLogger(__name__)


# @function_tool
async def EmergencyCall(imageBase64: str) -> bool:
    """Process an emergency call with image analysis and audio notification.

    Args:
        imageBase64: Base64 encoded image string

    Returns:
        bool: True if emergency call was processed successfully, False otherwise
    """
    try:
        # Analyze the image
        img_analizer = ImgAnalizer()
        text = await img_analizer.analize_image(imageBase64)
        if not text:
            logger.error("Failed to analyze image")
            return False

        # Convert text to speech
        openai_service = OpenAIService()
        audio_data = await openai_service.text_to_speech(text)
        if not audio_data:
            logger.error("Failed to convert text to speech")
            return False

        # Make the emergency call
        caller = EmergencyCallHelper()
        success = caller.make_call("517736641", audio_data)

        if success:
            logger.info("Emergency call processed successfully")
        else:
            logger.error("Failed to process emergency call")

        return success

    except Exception as e:
        logger.exception("Error in emergency call processing: %s", e)
        return False


class EmergencyCallHelper:
    def __init__(self):
        self.phone_manager = ADBPhoneCallManager()

    def analize_image(self, image) -> bool:
        """Analyze the image and return True if it is an emergency call, False otherwise."""
        return True

    def make_call(self, phone_number: str, audio_data: bytes) -> bool:
        try:
            phone_manager = ADBPhoneCallManager()

            # Example phone number to call
            phone_number = "517736641"

            logger.info(f"Making call to {phone_number}...")
            if phone_manager.make_call(phone_number):
                logger.info("Call initiated successfully")

                # Wait for the call to become active
                logger.info("Waiting for call to become active...")
                if phone_manager.wait_for_answer(timeout=20):
                    logger.info("Call is active")

                    # Simulate a call duration
                    logger.info("Call in progress...")
                    time.sleep(2)
                    phone_manager.play_audio(audio_data)

                    # Check call status
                    status = phone_manager.get_call_status()
                    logger.info(f"Current call status: {status}")

                    # End the call
                    logger.info("Ending call...")
                    if phone_manager.end_call():
                        logger.info("Call ended successfully")
                    else:
                        logger.warning("Failed to end call")
                else:
                    phone_manager.end_call()
                    logger.warning("Call did not become active within timeout period")
            else:
                logger.error("Failed to initiate call")

        except RuntimeError as e:
            logger.error(f"Runtime error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")


if __name__ == "__main__":
    # Get the image path
    image_path = Path("static/Elderly-Falls.jpg")

    if not image_path.exists():
        logger.error("Test image not found at %s", image_path)
        exit(1)

    try:
        # Read and encode the image as base64
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode("utf-8")

        # Use asyncio to run the async function
        asyncio.run(EmergencyCall(base64_image))

    except Exception as e:
        logger.exception("Failed to process image: %s", e)
        exit(1)
