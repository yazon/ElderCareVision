import requests
from typing import Optional
import json
import random
import string

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
        cmd = "Login"
        login_url = f"{self.base_url}/api.cgi?cmd={cmd}"
        
        login_data = [{
            "cmd": f"{cmd}",
            "action": 0,
            "param": {
                "User": {
                    "Version": 0,
                    "userName": self.username,
                    "password": self.password
                }
            }
        }]
        
        try:
            response = self.session.post(login_url, json=login_data)
            response.raise_for_status()
            
            data = response.json()
            if data[0]["code"] == 0:  # Success code
                self.token = data[0]["value"]["Token"]["name"]
            else:
                raise Exception(f"Login failed: {data[0]['code']}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to connect to camera: {str(e)}")
    
    def _make_request(self, cmd: str, params: dict | None = None) -> dict | bytes:
        """
        Make an authenticated request to the Reolink camera.
        
        Args:
            cmd (str): The command to execute
            params (dict | None): Additional parameters for the command
            
        Returns:
            dict | bytes: The response from the camera, either JSON data or binary data for snapshots
        """
        if not self.token:
            raise Exception("Not logged in")
            
        request_data = [{
            "cmd": cmd,
            "param": params or {},
        }]
        
        request_url = f"{self.base_url}/api.cgi?cmd={request_data[0]['cmd']}&token={self.token}"
        if params is not None:
            for key, value in params.items():
                request_url += f"&{key}={value}"
        
        try:
            response = self.session.post(request_url, json=request_data)
            response.raise_for_status()
            
            # For snapshot command, return the raw binary data
            if cmd == "Snap":
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
        return self._make_request("GetDevInfo")
    
    def get_ability_info(self) -> dict:
        """
        Get device information from the camera.
        
        Returns:
            dict: Device information
        """
        return self._make_request("GetAbility")
    
    def get_network_info(self) -> dict:
        """
        Get network information from the camera.
        
        Returns:
            dict: Network information
        """
        return self._make_request("GetNetPort")
    
    def make_snapshot(self) -> bytes:
        """
        Take a snapshot from the camera.
        
        Returns:
            bytes: Binary image data
        """
        length = 10
        param_dict = dict()
        param_dict["channel"] = 0
        param_dict["rs"] = "".join(random.choices(string.ascii_letters + string.digits, k=length))
        return self._make_request("Snap", param_dict)

    def logout(self) -> None:
        """
        Logout from the camera and clear the session.
        """
        if self.token:
            self._make_request("Logout")
            self.token = None
            self.session.close() 

if __name__ == "__main__":
    reolink = ReolinkAPI("192.168.1.100", "admin", "")
    print(reolink.get_device_info())
    print(reolink.get_network_info())
    print(reolink.get_ability_info())
    
    # Save snapshot to a file
    snapshot_data = reolink.make_snapshot()
    with open("snapshot.jpg", "wb") as f:
        f.write(snapshot_data)
    
    reolink.logout()