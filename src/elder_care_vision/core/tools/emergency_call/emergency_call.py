"""Example usage of ADBPhoneCallManager for making and managing phone calls."""

import logging

from elder_care_vision.config.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


async def emergency_call_tool() -> None:
    """Make an emergency call."""
    logger.info("Making emergency call...")
    # emergency = EmergencyCall()
    # emergency.make_call("517736641")


# @function_tool
# def EmergencyCall(imageBase64: str) -> bool:
#     """Analyze the image and return True if it is an emergency call, False otherwise."""
#     return True


# class EmergencyCall:
#     def __init__(self):
#         self.phone_manager = ADBPhoneCallManager()

#     def analize_image(self, image) -> bool:
#         """Analyze the image and return True if it is an emergency call, False otherwise."""
#         return True

#     def make_call(self, phone_number: str) -> bool:
#         try:
#             phone_manager = ADBPhoneCallManager()

#             # Example phone number to call
#             phone_number = "517736641"

#             logger.info(f"Making call to {phone_number}...")
#             if phone_manager.make_call(phone_number):
#                 logger.info("Call initiated successfully")

#                 # Wait for the call to become active
#                 logger.info("Waiting for call to become active...")
#                 if phone_manager.wait_for_answer(timeout=20):
#                     logger.info("Call is active")

#                     # Simulate a call duration
#                     logger.info("Call in progress...")
#                     time.sleep(10)  # Simulate 10 seconds of call time

#                     # Check call status
#                     status = phone_manager.get_call_status()
#                     logger.info(f"Current call status: {status}")

#                     # End the call
#                     logger.info("Ending call...")
#                     if phone_manager.end_call():
#                         logger.info("Call ended successfully")
#                     else:
#                         logger.warning("Failed to end call")
#                 else:
#                     phone_manager.end_call()
#                     logger.warning("Call did not become active within timeout period")
#             else:
#                 logger.error("Failed to initiate call")

#         except RuntimeError as e:
#             logger.error(f"Runtime error: {e}")
#         except Exception as e:
#             logger.exception(f"Unexpected error: {e}")


# def main():
#     """Demonstrate the usage of ADBPhoneCallManager."""
#     emergency = EmergencyCall()
#     emergency.make_call("517736641")


# if __name__ == "__main__":
#     main()
