"""Example usage of ADBPhoneCallManager for making and managing phone calls."""

import logging

from elder_care_vision.config.logging_config import setup_logging
from elder_care_vision.core.tools.emergency_call.adb_phone_call_manager import ADBPhoneCallManager
from elder_care_vision.core.tools.emergency_call.img_analizer import ImgAnalizer

setup_logging()

logger = logging.getLogger(__name__)


@function_tool
def EmergencyCall(imageBase64: str) -> bool:
    img_analizer = ImgAnalizer()
    text = img_analizer.analize_image(imageBase64)
    if text == "":
        return False

    caller = EmergencyCallHelper()
    caller.make_call("517736641")


class EmergencyCallHelper:
    def __init__(self):
        self.phone_manager = ADBPhoneCallManager()

    def analize_image(self, image) -> bool:
        """Analyze the image and return True if it is an emergency call, False otherwise."""
        return True

    def make_call(self, phone_number: str) -> bool:
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
                    time.sleep(10)  # Simulate 10 seconds of call time

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
