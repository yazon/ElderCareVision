"""Emergency Call System Module

This module provides functionality for handling emergency situations through automated calls and SMS notifications.
It integrates image analysis, text-to-speech conversion, and phone communication capabilities to provide a comprehensive
emergency response system.

The system works by:
1. Analyzing images for emergency situations
2. Generating appropriate notification messages
3. Converting messages to speech
4. Making emergency calls with audio messages
5. Sending SMS notifications as backup

Example:
    ```python
    from elder_care_vision.core.tools.emergency_call import emergency_call_tool

    # Process an emergency situation
    success = await emergency_call_tool(fall_detection_result, health_status)
    ```
"""

import base64
import logging
import asyncio
import time
from pathlib import Path

from elder_care_vision.config.logging_config import setup_logging
from elder_care_vision.core.tools.emergency_call.adb_phone_call_manager import ADBPhoneCallManager
from elder_care_vision.core.tools.emergency_call.img_analizer import ImgAnalizer
from elder_care_vision.services.openai_service import OpenAIService
from elder_care_vision.utils.utils import load_config
from elder_care_vision.core.agents.psa_data import FallDetectionResult

setup_logging()

logger = logging.getLogger(__name__)


async def emergency_call_tool(fall_detection_result: FallDetectionResult, health_status: str) -> bool:
    """Process an emergency call with image analysis and audio notification.

    This function handles the complete emergency response process:
    1. Analyzes the fall detection result
    2. Generates appropriate messages based on health status
    3. Sends SMS notifications to all configured contacts
    4. Makes emergency calls with audio messages
    5. Handles any errors that occur during the process

    Args:
        fall_detection_result: The result of fall detection analysis
        health_status: Current health status of the patient ("not_ok" or "needs_help")

    Returns:
        bool: True if emergency call was processed successfully, False otherwise

    Example:
        ```python
        result = FallDetectionResult(...)
        success = await emergency_call_tool(result, "needs_help")
        ```
    """
    try:
        story = FallDetectionResult.analysis

        if health_status == "not_ok":
            story += "Parient is not responding"
        elif health_status == "needs_help":
            story += "Patient is responding but needs help"

        # Get config
        config = load_config()["emergency_call"]

        for contact in config["contacts"]:
            # Analyze the image
            img_analizer = ImgAnalizer()
            imageBase64 = fall_detection_result.fall_image
            text = await img_analizer.analize_image(imageBase64, contact["relationship"], story)
            if not text:
                logger.error("Failed to analyze image")
                continue

            # Send SMS
            caller = EmergencyCallHelper()
            success = caller.send_sms(contact["phone_number"], text)
            if not success:
                logger.error("Failed to send SMS")

            # Convert text to speech
            openai_service = OpenAIService()
            audio_data = await openai_service.text_to_speech(text)
            if not audio_data:
                logger.error("Failed to convert text to speech")
                continue

            # Make the emergency call
            try:
                success = caller.make_call(contact["phone_number"], audio_data)
                if success:
                    logger.info("Emergency call processed successfully")
                    return True
                else:
                    logger.error("Failed to process emergency call")
                    continue
            except Exception as e:
                logger.error("Error in emergency call processing: %s", str(e))
                continue

        return False

    except Exception as e:
        logger.error("Error in emergency call processing: %s", str(e))
        return False


class EmergencyCallHelper:
    """Helper class for managing emergency calls and SMS notifications.

    This class provides methods for:
    - Making emergency phone calls
    - Sending SMS notifications
    - Playing audio messages during calls
    - Managing call status and duration

    Attributes:
        phone_manager: Instance of ADBPhoneCallManager for handling phone operations

    Example:
        ```python
        helper = EmergencyCallHelper()
        success = helper.make_call("1234567890", audio_data)
        ```
    """

    def __init__(self):
        """Initialize the EmergencyCallHelper.

        Raises:
            ValueError: If no Android device is found
        """
        try:
            self.phone_manager = ADBPhoneCallManager()
        except ValueError as e:
            logger.error("No Android device found. Please connect a device and try again.")
            raise

    def make_call(self, phone_number: str, audio_data: bytes) -> bool:
        """Make an emergency call with audio message.

        This method:
        1. Initiates a call to the specified number
        2. Waits for the call to be answered
        3. Plays the audio message
        4. Monitors call status
        5. Ends the call when done

        Args:
            phone_number: The phone number to call
            audio_data: The audio message to play during the call

        Returns:
            bool: True if call was successful, False otherwise

        Example:
            ```python
            success = helper.make_call("1234567890", audio_data)
            ```
        """
        if self.phone_manager is None:
            logger.error("No phone manager available to make a call")
            return False
        try:
            phone_manager = ADBPhoneCallManager()

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
                        return True
                    else:
                        logger.warning("Failed to end call")
                else:
                    phone_manager.end_call()
                    logger.warning("Call did not become active within timeout period")
            else:
                logger.error("Failed to initiate call")

            return False

        except RuntimeError as e:
            logger.error("Runtime error occurred during call")
            return False
        except Exception as e:
            logger.error("Unexpected error occurred during call")
            return False

    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send an SMS notification.

        Args:
            phone_number: The phone number to send the SMS to
            message: The message text to send

        Returns:
            bool: True if SMS was sent successfully, False otherwise

        Example:
            ```python
            success = helper.send_sms("1234567890", "Emergency notification")
            ```
        """
        if self.phone_manager is None:
            logger.error("No phone manager available to send SMS")
            return False
        return self.phone_manager.send_sms(phone_number, message)


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
        success = asyncio.run(emergency_call_tool(base64_image))
        exit(0 if success else 1)

    except Exception as e:
        logger.error("Failed to process image: %s", str(e))
        exit(1)
