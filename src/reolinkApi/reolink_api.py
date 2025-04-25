import requests
from typing import Optional
import json
import random
import string
import time
from datetime import datetime

class ReolinkAPI:
    def __init__(self, ip: str, username: str, password: str):
        """
        Initialize the Reolink API client and perform login.
        
        Args:
            ip (str): The IP address of the Reolink camera
            username (str): The username for authentication
            password (str): The password for authentication
        """
        self.base_url = f"http://{ip}"
        self.username = username
        self.password = password
        self.token = None
        self.session = requests.Session()
        
        # Perform login during initialization
        self.login()
    
    def login(self) -> None:
        """
        Perform login to the Reolink camera and store the authentication token.
        """
        url_params = {"cmd": "Login"}
        post_data = {
            "User": {
                "Version": 0,
                "userName": self.username,
                "password": self.password
            }
        }
        
        try:
            response = self.session.post(f"{self.base_url}/api.cgi?cmd=Login", json=[{
                "cmd": "Login",
                "action": 0,
                "param": post_data
            }])
            response.raise_for_status()
            
            data = response.json()
            if data[0]["code"] == 0:  # Success code
                self.token = data[0]["value"]["Token"]["name"]
            else:
                raise Exception(f"Login failed: {data[0]['code']}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to connect to camera: {str(e)}")
    
    def _make_request(self, url_params: dict, post_data: dict | None = None) -> dict | bytes:
        """
        Make an authenticated request to the Reolink camera.
        
        Args:
            url_params (dict): URL parameters including command and token
            post_data (dict | None): POST request data dictionary. Defaults to None.
            
        Returns:
            dict | bytes: The response from the camera, either JSON data or binary data for snapshots
        """
        if not self.token:
            raise Exception("Not logged in")
            
        # Build URL with parameters
        request_url = f"{self.base_url}/api.cgi"
        if url_params:
            request_url += "?" + "&".join(f"{key}={value}" for key, value in url_params.items())
        
        try:
            response = self.session.post(request_url, json=post_data)
            response.raise_for_status()
            
            # For snapshot command, return the raw binary data
            if url_params.get("cmd") == "Snap":
                return response.content
            
            # For other commands, parse as JSON
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def get_device_info(self) -> dict:
        """
        Get device information from the camera.
        
        Returns:
            dict: Device information
        """
        url_params = {"cmd": "GetDevInfo", "token": self.token}
        request_data = [{"cmd": "GetDevInfo"}]
        return self._make_request(url_params, request_data)
    
    def get_ability_info(self) -> dict:
        """
        Get device information from the camera.
        
        Returns:
            dict: Device information
        """
        url_params = {"cmd": "GetAbility", "token": self.token}
        request_data = [{"cmd": "GetAbility"}]
        return self._make_request(url_params, request_data)
    
    def get_network_info(self) -> dict:
        """
        Get network information from the camera.
        
        Returns:
            dict: Network information
        """
        url_params = {"cmd": "GetNetPort", "token": self.token}
        request_data = [{"cmd": "GetNetPort"}]
        return self._make_request(url_params, request_data)
    
    def make_snapshot(self) -> bytes:
        """
        Take a snapshot from the camera.
        
        Returns:
            bytes: Binary image data
        """
        length = 10
        url_params = {
            "cmd": "Snap",
            "token": self.token,
            "channel": "0",
            "rs": "".join(random.choices(string.ascii_letters + string.digits, k=length))
        }
        request_data = [{"cmd": "Snap", "channel": 0}]
        return self._make_request(url_params, request_data)

    def get_motion_state(self) -> dict:
        """
        Get the current motion detection state from the camera.
        
        Returns:
            dict: Motion detection state information including:
                - Motion detection status
                - Sensitivity settings
                - Detection areas
                - Schedule settings
        """
        url_params = {"cmd": "GetMdState", "token": self.token}
        request_data = [{"cmd": "GetMdState"}]
        return self._make_request(url_params, request_data)

    def get_ai_state(self) -> dict:
        """
        Get the current AI detection state from the camera.
        
        Returns:
            dict: AI detection state information including:
                - AI detection status
                - Detection types (person, vehicle, pet, etc.)
                - Sensitivity settings
                - Detection areas
        """
        url_params = {"cmd": "GetAiState", "token": self.token}
        request_data = [{"cmd": "GetAiState"}]
        return self._make_request(url_params, request_data)

    def set_ai_config(self, config: dict) -> dict:
        """
        Configure AI detection settings on the camera.
        
        Args:
            config (dict): Configuration parameters including:
                - enable (bool): Enable/disable AI detection
                - types (list): List of detection types to enable
                - sensitivity (int): Detection sensitivity level
                - areas (list): Detection areas configuration
        
        Returns:
            dict: Response from the camera indicating success or failure
        """
        url_params = {"cmd": "SetAiCfg", "token": self.token}
        request_data = [{"cmd": "SetAiCfg", "action":0,"param": config, "channel":0}]
        return self._make_request(url_params, request_data)

    def send_heart_beat(self) -> dict:
        """
        Send a heartbeat signal to keep the connection alive with the camera.
        This should be called periodically to maintain the session.
        
        Returns:
            dict: Response from the camera indicating the connection is alive
        """
        url_params = {"cmd": "HeartBeat", "token": self.token}
        request_data = [{"cmd": "HeartBeat"}]
        return self._make_request(url_params, request_data)

    def set_ptz_ctrl(self, command: str, channel: int = 0, speed: int = 1) -> dict:
        """
        Control the PTZ (Pan-Tilt-Zoom) functionality of the camera.
        
        Args:
            command (str): PTZ command to execute (e.g., "Left", "Right", "Up", "Down", "ZoomIn", "ZoomOut")
            channel (int, optional): Camera channel to control. Defaults to 0.
            speed (int, optional): Movement speed (1-8). Defaults to 1.
        
        Returns:
            dict: Response from the camera indicating success or failure
        """
        url_params = {
            "cmd": "PtzCtrl",
            "token": self.token
        }
        request_data = [{"cmd": "PtzCtrl", "param": {"channel": channel, "op": command, "speed": speed}}]
        return self._make_request(url_params, request_data)

    def logout(self) -> None:
        """
        Logout from the camera and clear the session.
        """
        if self.token:
            url_params = {"cmd": "Logout", "token": self.token}
            request_data = [{"cmd": "Logout"}]
            self._make_request(url_params, request_data)
            self.token = None
            self.session.close()


def camera_patrol(reolink, patrol_time):
    start_time = time.time()
    direction = "Right"
    while(time.time() - start_time < patrol_time):
        for i in range(10):
            reolink.set_ptz_ctrl(command=direction, speed=3)
            time.sleep(.2)
            reolink.set_ptz_ctrl(command="Stop")
            time.sleep(1)
            print(reolink.get_motion_state()[0]['value'])
            print(reolink.get_ai_state()[0]['value'])
        
            # Save snapshot with timestamp including milliseconds
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Last 3 digits of microseconds = milliseconds
            snapshot_data = reolink.make_snapshot()
            filename = f"snapshot_{timestamp}.png"
            st_time = time.time()
            with open(filename, "wb") as f:
                f.write(snapshot_data)
            print(f"time spend to take a picture {time.time() - st_time}")
        if direction == 'Right':
            direction = 'Left'
        else:
            direction = "Right"

def camera_burst_snapshot(reolink, burst):
    print("Save burst")
    for _ in range(burst):
        # Save snapshot with timestamp including milliseconds
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Last 3 digits of microseconds = milliseconds
        snapshot_data = reolink.make_snapshot()
        filename = f"snapshot_{timestamp}.png"
        st_time = time.time()
        with open(filename, "wb") as f:
            f.write(snapshot_data)
    
if __name__ == "__main__":
    reolink = ReolinkAPI("192.168.1.100", "admin", "")
    print(reolink.get_device_info())
    print(reolink.get_network_info())
    print(reolink.get_ability_info())
    print(reolink.get_motion_state())
    print(reolink.get_ai_state())
    print(reolink.send_heart_beat())

    print(reolink.set_ptz_ctrl(command="Left", speed=3))
                
    time.sleep(1)
    ai_config_params = {"aiTrack": 1,
        "trackType": {
            "dog_cat" : 0,
            "face" : 1,
            "people" : 1,
            "vehicle" : 0},
        "AiDetectType": {
            "people": 1,
            "vehicle": 0,
            "dog_cat": 0,
            "face": 1
        }}
    print(reolink.set_ai_config(ai_config_params))

    # camera_patrol(reolink, 1)

    camera_burst_snapshot(reolink, 100)

    reolink.logout()