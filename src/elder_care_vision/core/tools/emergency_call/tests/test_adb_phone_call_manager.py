"""Tests for the ADBPhoneCallManager class."""

from unittest.mock import patch

import pytest

from elder_care_vision.call_system.call_system import ADBPhoneCallManager, CallStatus


@pytest.fixture
def mock_adb():
    """Create a mock ADB instance."""
    with patch("elder_care_vision.call_system.call_system.ADB") as mock:
        yield mock


@pytest.fixture
def phone_manager(mock_adb):
    """Create an ADBPhoneCallManager instance with mocked ADB."""
    return ADBPhoneCallManager()


def test_init_with_device_id(mock_adb):
    """Test initialization with a specific device ID."""
    device_id = "test_device"
    manager = ADBPhoneCallManager(device_id=device_id)
    mock_adb.return_value.set_target_device.assert_called_once_with(device_id)


def test_init_without_device_id(mock_adb):
    """Test initialization without device ID."""
    mock_adb.return_value.get_devices.return_value = ["device1"]
    manager = ADBPhoneCallManager()
    mock_adb.return_value.set_target_device.assert_called_once_with("device1")


def test_init_no_devices(mock_adb):
    """Test initialization when no devices are connected."""
    mock_adb.return_value.get_devices.return_value = []
    with pytest.raises(RuntimeError, match="No ADB devices connected"):
        ADBPhoneCallManager()


def test_make_call_success(phone_manager, mock_adb):
    """Test successful call initiation."""
    phone_number = "1234567890"
    assert phone_manager.make_call(phone_number) is True
    mock_adb.return_value.shell.assert_called_once_with(f"am start -a android.intent.action.CALL -d tel:{phone_number}")


def test_make_call_invalid_number(phone_manager):
    """Test call initiation with invalid phone number."""
    with pytest.raises(ValueError, match="Invalid phone number"):
        phone_manager.make_call("")


def test_make_call_failure(phone_manager, mock_adb):
    """Test failed call initiation."""
    mock_adb.return_value.shell.side_effect = Exception("ADB error")
    assert phone_manager.make_call("1234567890") is False


def test_end_call_success(phone_manager, mock_adb):
    """Test successful call termination."""
    assert phone_manager.end_call() is True
    mock_adb.return_value.shell.assert_called_once_with("input keyevent KEYCODE_ENDCALL")


def test_end_call_failure(phone_manager, mock_adb):
    """Test failed call termination."""
    mock_adb.return_value.shell.side_effect = Exception("ADB error")
    assert phone_manager.end_call() is False


def test_get_call_status_idle(phone_manager, mock_adb):
    """Test getting IDLE call status."""
    mock_adb.return_value.shell.return_value = "mCallState=0"
    assert phone_manager.get_call_status() == CallStatus.IDLE


def test_get_call_status_ringing(phone_manager, mock_adb):
    """Test getting RINGING call status."""
    mock_adb.return_value.shell.return_value = "mCallState=1"
    assert phone_manager.get_call_status() == CallStatus.RINGING


def test_get_call_status_active(phone_manager, mock_adb):
    """Test getting ACTIVE call status."""
    mock_adb.return_value.shell.return_value = "mCallState=2"
    assert phone_manager.get_call_status() == CallStatus.ACTIVE


def test_get_call_status_error(phone_manager, mock_adb):
    """Test getting ERROR call status."""
    mock_adb.return_value.shell.return_value = "mCallState=3"
    assert phone_manager.get_call_status() == CallStatus.ERROR


def test_get_call_status_exception(phone_manager, mock_adb):
    """Test getting call status when ADB command fails."""
    mock_adb.return_value.shell.side_effect = Exception("ADB error")
    assert phone_manager.get_call_status() == CallStatus.ERROR


def test_wait_for_call_status_success(phone_manager, mock_adb):
    """Test successful wait for call status."""
    mock_adb.return_value.shell.return_value = "mCallState=2"
    assert phone_manager.wait_for_call_status(CallStatus.ACTIVE, timeout=1) is True


def test_wait_for_call_status_timeout(phone_manager, mock_adb):
    """Test timeout while waiting for call status."""
    mock_adb.return_value.shell.return_value = "mCallState=0"
    assert phone_manager.wait_for_call_status(CallStatus.ACTIVE, timeout=1) is False


def test_wait_for_call_status_exception(phone_manager, mock_adb):
    """Test wait for call status when ADB command fails."""
    mock_adb.return_value.shell.side_effect = Exception("ADB error")
    assert phone_manager.wait_for_call_status(CallStatus.ACTIVE, timeout=1) is False
