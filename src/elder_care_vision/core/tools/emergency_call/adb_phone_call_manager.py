"""Module for handling ADB calls and call status monitoring."""

import time
import logging
from enum import Enum
from typing import Optional

from ppadb.client import Client as AdbClient
from elder_care_vision.config.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

class CallStatus(Enum):
    """Enum representing different call states."""
    IDLE = "idle"
    DIALING = "dialing"
    RINGING = "ringing"
    ACTIVE = "active"
    ENDED = "ended"
    DECLINED = "declined"
    ERROR = "error"


class ADBPhoneCallManager:
    """Class for managing phone calls through ADB on Android devices."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5037, device_index: int = 0) -> None:
        """Initialize the ADBPhoneCallManager."""
        self.client = AdbClient(host=host, port=port)
        self.device = self.client.devices()[device_index]
        self._last_status = CallStatus.IDLE

    def make_call(self, phone_number: str) -> bool:
        """Make a phone call using ADB."""
        if not phone_number or not phone_number.strip():
            raise ValueError("Invalid phone number")

        try:
            self.device.shell(f"am start -a android.intent.action.CALL -d tel:{phone_number}")
            self._last_status = CallStatus.IDLE
            logger.info(f"Call initiated to {phone_number}")
            return True
        except Exception as e:
            logger.error(f"Failed to initiate call: {e}")
            return False

    def end_call(self) -> bool:
        """End the current call."""
        try:
            self.device.shell("input keyevent KEYCODE_ENDCALL")
            logger.info("Call ended.")
            return True
        except Exception as e:
            logger.error(f"Failed to end call: {e}")
            return False

    def get_call_status(self) -> CallStatus:
        """Get the current call status."""
        try:
            output = self.device.shell("dumpsys telephony.registry")
            for line in output.splitlines():
                if "mForegroundCallState" in line:
                    logger.debug(f"Call state line: {line}")
                    state_code = line.strip().split("=")[-1]
                    current_status = self._last_status

                    if state_code == "0":
                        current_status = CallStatus.IDLE
                    elif state_code == "1":
                        current_status = CallStatus.ACTIVE
                    elif state_code == "3":
                        current_status = CallStatus.DIALING
                    elif state_code == "4":
                        current_status = CallStatus.DIALING

                    self._last_status = current_status
                    return current_status

        except Exception as e:
            logger.error(f"Error getting call status: {e}")
            return CallStatus.ERROR

    def wait_for_answer(self, timeout: int = 60) -> bool:
        """Wait for the call to be answered or declined."""
        prev_status = None
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_call_status()
            logger.debug(f"Current call status: {status}")
            if status == CallStatus.ACTIVE:
                logger.info("Call answered.")
                return True
            if prev_status == CallStatus.DIALING and status == CallStatus.IDLE:
                logger.info("Call declined or ended.")
                return False
            prev_status = status
            time.sleep(1)
        logger.info("Call answer timeout reached.")
        return False
