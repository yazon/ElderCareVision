"""Android Device Communication Module

This module provides functionality for managing phone calls and SMS messages through ADB (Android Debug Bridge).
It enables automated phone operations including making calls, sending SMS, and playing audio messages on Android devices.

Key Features:
- Phone call management (make, end, monitor)
- SMS sending with message splitting
- Audio playback through computer speakers
- Call status monitoring
- Error handling and logging

Example:
    ```python
    from elder_care_vision.core.tools.emergency_call import ADBPhoneCallManager

    # Initialize the manager
    manager = ADBPhoneCallManager()

    # Make a call
    success = manager.make_call("1234567890")

    # Send an SMS
    success = manager.send_sms("1234567890", "Emergency message")
    ```
"""

import io
import logging
import textwrap
import time
from enum import Enum

import sounddevice as sd
import soundfile as sf
from ppadb.client import Client as AdbClient

from elder_care_vision.config.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


class CallStatus(Enum):
    """Enum representing different call states.

    States:
        IDLE: No active call
        DIALING: Call is being initiated
        RINGING: Call is ringing
        ACTIVE: Call is connected
        ENDED: Call has ended
        DECLINED: Call was declined
        ERROR: Error occurred
    """

    IDLE = "idle"
    DIALING = "dialing"
    RINGING = "ringing"
    ACTIVE = "active"
    ENDED = "ended"
    DECLINED = "declined"
    ERROR = "error"


class ADBPhoneCallManager:
    """Class for managing phone calls through ADB on Android devices.

    This class provides methods for:
    - Making and ending phone calls
    - Monitoring call status
    - Sending SMS messages
    - Playing audio messages

    The class uses ADB to communicate with Android devices and requires:
    - ADB server running
    - Android device connected
    - USB debugging enabled on device

    Attributes:
        client: ADB client instance
        device: Connected Android device
        _last_status: Last known call status
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5037, device_index: int = 0) -> None:
        """Initialize the ADBPhoneCallManager.

        Args:
            host: ADB server host address
            port: ADB server port
            device_index: Index of device to use if multiple devices are connected

        Raises:
            ValueError: If no Android devices are found
        """
        self.client = AdbClient(host=host, port=port)
        if self.client.devices():
            self.device = self.client.devices()[device_index]
        else:
            raise ValueError("No devices found")
        self._last_status = CallStatus.IDLE

    def make_call(self, phone_number: str) -> bool:
        """Make a phone call using ADB.

        This method initiates a phone call to the specified number using Android's
        telephony system through ADB commands.

        Args:
            phone_number: The phone number to call

        Returns:
            bool: True if call was initiated successfully, False otherwise

        Raises:
            ValueError: If phone number is invalid

        Example:
            ```python
            success = manager.make_call("1234567890")
            ```
        """
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
        """End the current call.

        This method sends the end call command to the Android device using ADB.

        Returns:
            bool: True if call was ended successfully, False otherwise

        Example:
            ```python
            success = manager.end_call()
            ```
        """
        try:
            self.device.shell("input keyevent KEYCODE_ENDCALL")
            logger.info("Call ended.")
            return True
        except Exception as e:
            logger.error(f"Failed to end call: {e}")
            return False

    def get_call_status(self) -> CallStatus:
        """Get the current call status.

        This method queries the Android device's telephony system to determine
        the current state of any active call.

        Returns:
            CallStatus: Current state of the call

        Example:
            ```python
            status = manager.get_call_status()
            if status == CallStatus.ACTIVE:
                print("Call is active")
            ```
        """
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
                    elif state_code == "3" or state_code == "4":
                        current_status = CallStatus.DIALING

                    self._last_status = current_status
                    return current_status

        except Exception as e:
            logger.error(f"Error getting call status: {e}")
            return CallStatus.ERROR

    def wait_for_answer(self, timeout: int = 60) -> bool:
        """Wait for the call to be answered or declined.

        This method polls the call status until either:
        - The call is answered
        - The call is declined
        - The timeout is reached

        Args:
            timeout: Maximum time in seconds to wait for answer

        Returns:
            bool: True if call was answered, False otherwise

        Example:
            ```python
            if manager.wait_for_answer(timeout=30):
                print("Call was answered")
            else:
                print("Call was not answered")
            ```
        """
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

    def play_audio(self, audio_data: bytes) -> bool:
        """Play audio on the computer using sounddevice and soundfile.

        This method plays audio data through the computer's speakers or
        a connected Plantronics device if available.

        Args:
            audio_data: The audio data in bytes format (e.g., MP3, WAV)

        Returns:
            bool: True if audio played successfully, False otherwise

        Example:
            ```python
            with open("message.mp3", "rb") as f:
                audio_data = f.read()
            success = manager.play_audio(audio_data)
            ```
        """
        try:
            device_index = None
            for idx, device in enumerate(sd.query_devices()):
                if device["name"] and "plantronics" in device["name"].lower():
                    device_index = idx
                    break

            # Create a file-like object from the audio data
            audio_file = io.BytesIO(audio_data)

            # Read the audio data using soundfile
            data, samplerate = sf.read(audio_file)

            # Play the audio using sounddevice
            if device_index is not None:
                sd.play(data, samplerate, device=device_index)
            else:
                sd.play(data, samplerate)
            sd.wait()  # Wait until the audio is finished playing

            logger.info("Audio played successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            return False

    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send an SMS message using ADB.

        This method sends an SMS message to the specified phone number.
        Long messages are automatically split into multiple parts.

        Args:
            phone_number: The recipient's phone number
            message: The message text to send

        Returns:
            bool: True if message was sent successfully, False otherwise

        Raises:
            ValueError: If phone number or message is invalid

        Example:
            ```python
            success = manager.send_sms("1234567890", "Emergency message")
            ```
        """
        if not phone_number or not phone_number.strip():
            raise ValueError("Invalid phone number")
        if not message or not message.strip():
            raise ValueError("Invalid message")

        try:
            parts = textwrap.wrap(message, 160)

            for i, part in enumerate(parts):
                print(f"Sending part {i + 1}/{len(parts)}: {part}")

                # Prepare the ADB command
                command = f'service call isms 7 i32 0 s16 com.android.shell s16 {phone_number} s16 null s16 "{part}" s16 null s16 null s16 null'
                self.device.shell(command)

                print(command)

            logger.info(f"SMS sent to {phone_number}")
            return True

        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False
