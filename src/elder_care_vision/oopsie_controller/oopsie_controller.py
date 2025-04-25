"""The grand conductor of your fall detection drama! 🎭

This module implements the OopsieController - the mastermind behind your fall detection ecosystem.
It orchestrates the overly-sensitive detector (oopsie_alert) and the sensible filter (oopsie_nanny)
to create a balanced and effective fall detection system.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
import time
import json
from colorama import init, Fore, Style
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import threading
import io
import queue

from .oopsie_alert.oopsie_alert import FallDetector
from .oopsie_nanny.oopsie_nanny import ImageRecognizer

# Initialize colorama
init()

# Configure logging with colors
class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    
    def format(self, record):
        if record.levelno == logging.ERROR:
            record.msg = f"{Fore.RED}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{Fore.YELLOW}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.INFO:
            record.msg = f"{Fore.GREEN}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.DEBUG:
            record.msg = f"{Fore.CYAN}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

# Custom handler to send logs to tkinter
class TkinterLogHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s"))
        
    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # This sets the base log level

# Create queue for log messages
log_queue = queue.Queue()

# Add console handler with DEBUG level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Console will show all messages
console_handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

# Add tkinter handler with INFO level
tk_handler = TkinterLogHandler(log_queue)
tk_handler.setLevel(logging.INFO)  # Tkinter window will show INFO and above
logger.addHandler(tk_handler)

class OopsieController:
    """The main controller that orchestrates fall detection and verification.
    
    This class integrates the sensitive fall detector (OopsieAlert) with the rational
    verification system (OopsieNanny) to create a balanced fall detection system.
    
    Attributes:
        alert: The sensitive fall detector that triggers on potential falls
        nanny: The rational verifier that confirms if a fall is real
        warning_frames: Number of frames to show warning when fall is confirmed
        current_warning_frames: Counter for current warning display
        fall_confirmed: Whether the current detection has been confirmed
        is_processing_video: Whether the controller is processing a video
        last_llm_request_time: Timestamp of the last LLM request
        llm_cooldown: Minimum seconds between LLM requests
        last_pose_data: Store of the last processed pose data
        thresholds: Configuration values loaded from JSON
        update_counter: Counter for update frequency
        threshold_history: Dictionary to store threshold history
        max_history_length: Maximum length of threshold history
        fig_thresholds: Matplotlib figure for threshold history plots
        axs_thresholds: Matplotlib axes for threshold history plots
        fig_detection: Matplotlib figure for detection history plots
        ax_detection: Matplotlib axes for detection history plots
        plot_colors: Colors for threshold history plots
        detection_colors: Colors for detection history plots
        plot_lines: Dictionary to store plot lines
        plot_points: Dictionary to store plot points
        min_lines: Dictionary to store min lines
        max_lines: Dictionary to store max lines
        text_boxes: Dictionary to store text boxes
        detection_history: Dictionary to store detection history
        detection_lines: Dictionary to store detection lines
        error_counters: Dictionary to store error counters
    """
    
    def __init__(self):
        """Initialize the OopsieController with its components."""
        self.alert = FallDetector()
        self.nanny = ImageRecognizer()
        
        # Load thresholds from config file
        config_path = Path(__file__).parent / "config" / "thresholds.json"
        try:
            with open(config_path, "r") as f:
                self.thresholds = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load thresholds from config file: {str(e)}")
            # Set default values if config file fails to load
            self.thresholds = {
                "head_detection": {
                    "tilt_threshold": 2.0,
                    "position_threshold": 0.3,
                    "shoulder_ratio_threshold": 2.0,
                    "hip_ratio_threshold": 1.5
                },
                "pose_detection": {
                    "movement_threshold": 0.15
                },
                "llm": {
                    "cooldown_seconds": 5
                },
                "warning": {
                    "frames": 5
                },
                "auto_update": {
                    "enabled": False,
                    "min_confidence": 0.8,
                    "max_adjustment": 0.1,
                    "update_frequency": 5
                }
            }
            
        self.warning_frames = self.thresholds["warning"]["frames"]
        self.current_warning_frames = 0
        self.fall_confirmed = False
        self.is_processing_video = False
        self.last_llm_request_time = 0
        self.llm_cooldown = self.thresholds["llm"]["cooldown_seconds"]
        self.last_pose_data = None
        self.update_counter = 0  # Counter for update frequency
        
        # Initialize threshold history
        self.threshold_history = {
            "tilt": [],
            "position": [],
            "shoulder_ratio": [],
            "hip_ratio": []
        }
        self.max_history_length = 50  # Keep last 50 values
        
        # Initialize detection history
        self.detection_history = {
            "algorithm": [],  # Algorithm detection status (0 or 1)
            "llm": [],       # LLM detection status (0 or 1)
            "confirmed": []  # Final confirmed status (0 or 1)
        }
        
        # Initialize error counters
        self.error_counters = {
            "algorithm": 0,  # Total algorithm detections
            "llm": 0,       # Total LLM confirmations
            "confirmed": 0   # Total confirmed falls
        }
        
        # Initialize matplotlib figures with smaller sizes
        self.fig_thresholds, self.axs_thresholds = plt.subplots(2, 2, figsize=(8, 6))
        self.fig_detection, self.ax_detection = plt.subplots(figsize=(8, 3))
        
        # Configure threshold plots
        self.fig_thresholds.suptitle("Threshold History", fontsize=12)
        self.fig_thresholds.patch.set_facecolor('#1e1e1e')
        self.fig_thresholds.patch.set_alpha(0.8)
        
        # Configure detection plot
        self.fig_detection.suptitle("Detection Status", fontsize=12)
        self.fig_detection.patch.set_facecolor('#1e1e1e')
        self.fig_detection.patch.set_alpha(0.8)
        
        # Configure subplots with smaller fonts
        for ax in self.axs_thresholds.flat:
            ax.set_facecolor('#2d2d2d')
            ax.tick_params(colors='white', labelsize=8)
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
        # Configure detection plot with smaller fonts
        self.ax_detection.set_facecolor('#2d2d2d')
        self.ax_detection.tick_params(colors='white', labelsize=8)
        self.ax_detection.spines['bottom'].set_color('white')
        self.ax_detection.spines['top'].set_color('white')
        self.ax_detection.spines['left'].set_color('white')
        self.ax_detection.spines['right'].set_color('white')
        self.ax_detection.xaxis.label.set_color('white')
        self.ax_detection.yaxis.label.set_color('white')
        self.ax_detection.set_ylim(-0.1, 1.1)
        self.ax_detection.set_yticks([0, 1])
        self.ax_detection.set_yticklabels(['No Fall', 'Fall'], fontsize=8)
        
        # Set titles and labels for threshold plots with smaller fonts
        self.axs_thresholds[0, 0].set_title("Tilt Threshold", color='white', fontsize=10)
        self.axs_thresholds[0, 1].set_title("Position Threshold", color='white', fontsize=10)
        self.axs_thresholds[1, 0].set_title("Shoulder Ratio", color='white', fontsize=10)
        self.axs_thresholds[1, 1].set_title("Hip Ratio", color='white', fontsize=10)
        
        # Colors for each metric
        self.plot_colors = {
            "tilt": "#00ffff",  # Cyan
            "position": "#ffff00",  # Yellow
            "shoulder_ratio": "#ff00ff",  # Magenta
            "hip_ratio": "#00ff00"  # Green
        }
        
        # Colors for detection lines
        self.detection_colors = {
            "algorithm": "#ff0000",  # Red
            "llm": "#00ff00",        # Green
            "confirmed": "#0000ff"   # Blue
        }
        
        # Initialize plot lines and points
        self.plot_lines = {}
        self.plot_points = {}
        self.min_lines = {}
        self.max_lines = {}
        self.text_boxes = {}
        
        # Initialize detection lines
        self.detection_lines = {}
        
        # Initialize plot elements
        metrics = {
            "tilt": (0, 0),
            "position": (0, 1),
            "shoulder_ratio": (1, 0),
            "hip_ratio": (1, 1)
        }
        
        for metric, (row, col) in metrics.items():
            ax = self.axs_thresholds[row, col]
            # Create initial empty line
            self.plot_lines[metric], = ax.plot([], [], color=self.plot_colors[metric], linewidth=1.5)
            # Create initial point
            self.plot_points[metric], = ax.plot([], [], 'o', color=self.plot_colors[metric], markersize=4)
            # Create min/max lines
            self.min_lines[metric] = ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)
            self.max_lines[metric] = ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)
            # Create text box with smaller font
            self.text_boxes[metric] = ax.text(0.02, 0.98, "", transform=ax.transAxes, color='white',
                                            verticalalignment='top', fontsize=8,
                                            bbox=dict(facecolor='black', alpha=0.5))
        
        # Initialize detection lines with thinner lines
        for detector in ["algorithm", "llm", "confirmed"]:
            self.detection_lines[detector], = self.ax_detection.plot([], [], 
                                                                   color=self.detection_colors[detector],
                                                                   linewidth=1.5,
                                                                   label=detector.capitalize())
        
        # Add legend to detection plot with smaller font
        self.ax_detection.legend(loc='upper right', facecolor='#2d2d2d', edgecolor='white', 
                               labelcolor='white', fontsize=8)
        
        # Add error counter text to detection plot
        self.error_text = self.ax_detection.text(0.02, 0.02, "", transform=self.ax_detection.transAxes, 
                                               color='white', fontsize=8,
                                               bbox=dict(facecolor='black', alpha=0.5))
        
        # Set initial layout with tighter spacing
        plt.tight_layout(pad=1.0)
        
        # Create tkinter window in a separate thread
        self.plot_thread = threading.Thread(target=self._run_plot_window, daemon=True)
        self.plot_thread.start()
        
        # Track last values to avoid unnecessary updates
        self.last_values = {metric: None for metric in metrics}
        
        logger.info("OopsieController initialized and ready for fall detection")
        
    def _run_plot_window(self) -> None:
        """Run the tkinter window in a separate thread."""
        try:
            # Create tkinter window
            self.root = tk.Tk()
            self.root.title("Fall Detection Monitor")
            self.root.geometry("800x600")  # Reduced window size
            self.root.configure(bg='#1e1e1e')
            
            # Create main frame
            main_frame = tk.Frame(self.root, bg='#1e1e1e')
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create plot frames
            threshold_frame = tk.Frame(main_frame, bg='#1e1e1e')
            threshold_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
            
            detection_frame = tk.Frame(main_frame, bg='#1e1e1e')
            detection_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)
            
            # Create labels for plot images
            self.threshold_label = tk.Label(threshold_frame, bg='#1e1e1e')
            self.threshold_label.pack(fill=tk.BOTH, expand=True)
            
            self.detection_label = tk.Label(detection_frame, bg='#1e1e1e')
            self.detection_label.pack(fill=tk.BOTH, expand=True)
            
            # Start periodic updates
            self._update_plots()
            
            # Start the tkinter event loop
            self.root.mainloop()
            
        except Exception as e:
            logger.error(f"Error in plot window thread: {str(e)}")
            
    def _update_plots(self) -> None:
        """Update both plot images in the tkinter window."""
        try:
            # Update threshold plot
            buf_threshold = io.BytesIO()
            self.fig_thresholds.savefig(buf_threshold, format='png', facecolor='#1e1e1e', edgecolor='none')
            buf_threshold.seek(0)
            
            # Update detection plot
            buf_detection = io.BytesIO()
            self.fig_detection.savefig(buf_detection, format='png', facecolor='#1e1e1e', edgecolor='none')
            buf_detection.seek(0)
            
            # Convert to PIL Images
            threshold_image = Image.open(buf_threshold)
            detection_image = Image.open(buf_detection)
            
            # Convert to PhotoImages
            threshold_photo = ImageTk.PhotoImage(threshold_image)
            detection_photo = ImageTk.PhotoImage(detection_image)
            
            # Update the labels
            self.threshold_label.configure(image=threshold_photo)
            self.threshold_label.image = threshold_photo
            
            self.detection_label.configure(image=detection_photo)
            self.detection_label.image = detection_photo
            
            # Schedule next update
            if hasattr(self, 'root'):
                self.root.after(100, self._update_plots)
            
        except Exception as e:
            logger.error(f"Error updating plots: {str(e)}")
            
    def _update_detection_history(self, algorithm_detected: bool, llm_detected: bool, confirmed: bool) -> None:
        """Update the detection history with new values."""
        try:
            # Convert boolean values to float (1.0 or 0.0)
            algorithm_value = 1.0 if algorithm_detected else 0.0
            llm_value = 1.0 if llm_detected else 0.0
            confirmed_value = 1.0 if confirmed else 0.0
            
            # Add new values to history
            self.detection_history["algorithm"].append(algorithm_value)
            self.detection_history["llm"].append(llm_value)
            self.detection_history["confirmed"].append(confirmed_value)
            
            # Keep history length limited
            for key in self.detection_history:
                if len(self.detection_history[key]) > self.max_history_length:
                    self.detection_history[key].pop(0)
            
            # Update detection plot
            x_data = list(range(len(self.detection_history["algorithm"])))
            for detector in ["algorithm", "llm", "confirmed"]:
                self.detection_lines[detector].set_data(x_data, self.detection_history[detector])
            
            # Adjust axes limits
            self.ax_detection.relim()
            self.ax_detection.autoscale_view()
            
            # Force redraw of the plot
            self.fig_detection.canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating detection history: {str(e)}")
            
    def _draw_threshold_history(self) -> None:
        """Update the matplotlib plots with current threshold history."""
        try:
            # Only update every 5 frames to reduce overhead
            self.update_counter += 1
            if self.update_counter % 5 != 0:
                return
                
            # Check if any values have changed
            values_changed = False
            for metric, history in self.threshold_history.items():
                if history and history[-1] != self.last_values[metric]:
                    values_changed = True
                    self.last_values[metric] = history[-1]
            
            if not values_changed:
                return
                
            # Update plots for each metric
            for metric, history in self.threshold_history.items():
                if not history:
                    continue
                    
                # Update line data
                x_data = list(range(len(history)))
                self.plot_lines[metric].set_data(x_data, history)
                
                # Update point
                self.plot_points[metric].set_data([x_data[-1]], [history[-1]])
                
                # Update min/max lines
                min_val = min(history)
                max_val = max(history)
                self.min_lines[metric].set_ydata([min_val, min_val])
                self.max_lines[metric].set_ydata([max_val, max_val])
                
                # Update text
                self.text_boxes[metric].set_text(f"Current: {history[-1]:.2f}")
                
                # Adjust axes limits
                ax = self.plot_lines[metric].axes
                ax.relim()
                ax.autoscale_view()
            
        except Exception as e:
            logger.error(f"Error updating threshold history plots: {str(e)}")
            
    def draw_poi(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Draw Point of Interest markers and calculation points on detected persons.
        
        Args:
            frame: The video frame to draw on
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Frame with all calculation points and thresholds drawn
        """
        if landmarks is None:
            return frame
            
        height, width = frame.shape[:2]
        
        # Get all key points used in calculations
        left_shoulder = landmarks.landmark[self.alert.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.alert.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[self.alert.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.alert.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Get head points (nose and ears)
        nose = landmarks.landmark[self.alert.mp_pose.PoseLandmark.NOSE]
        left_ear = landmarks.landmark[self.alert.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks.landmark[self.alert.mp_pose.PoseLandmark.RIGHT_EAR]
        
        # Convert normalized coordinates to pixel coordinates
        ls_x = int(left_shoulder.x * width)
        ls_y = int(left_shoulder.y * height)
        rs_x = int(right_shoulder.x * width)
        rs_y = int(right_shoulder.y * height)
        lh_x = int(left_hip.x * width)
        lh_y = int(left_hip.y * height)
        rh_x = int(right_hip.x * width)
        rh_y = int(right_hip.y * height)
        
        # Convert head points to pixel coordinates
        nose_x = int(nose.x * width)
        nose_y = int(nose.y * height)
        le_x = int(left_ear.x * width)
        le_y = int(left_ear.y * height)
        re_x = int(right_ear.x * width)
        re_y = int(right_ear.y * height)
        
        # Calculate center points
        shoulder_center_x = (ls_x + rs_x) // 2
        shoulder_center_y = (ls_y + rs_y) // 2
        hip_center_x = (lh_x + rh_x) // 2
        hip_center_y = (lh_y + rh_y) // 2
        
        # Calculate head center and orientation
        head_center_x = (le_x + re_x) // 2
        head_center_y = (le_y + re_y) // 2
        
        # Draw shoulder points and line
        cv2.circle(frame, (ls_x, ls_y), 5, (0, 255, 255), -1)  # Left shoulder (cyan)
        cv2.circle(frame, (rs_x, rs_y), 5, (0, 255, 255), -1)  # Right shoulder (cyan)
        cv2.line(frame, (ls_x, ls_y), (rs_x, rs_y), (0, 255, 255), 2)  # Shoulder line
        
        # Draw hip points and line
        cv2.circle(frame, (lh_x, lh_y), 5, (255, 255, 0), -1)  # Left hip (yellow)
        cv2.circle(frame, (rh_x, rh_y), 5, (255, 255, 0), -1)  # Right hip (yellow)
        cv2.line(frame, (lh_x, lh_y), (rh_x, rh_y), (255, 255, 0), 2)  # Hip line
        
        # Draw head points and line
        cv2.circle(frame, (nose_x, nose_y), 5, (255, 0, 0), -1)  # Nose (blue)
        cv2.circle(frame, (le_x, le_y), 5, (255, 0, 0), -1)  # Left ear (blue)
        cv2.circle(frame, (re_x, re_y), 5, (255, 0, 0), -1)  # Right ear (blue)
        cv2.line(frame, (le_x, le_y), (re_x, re_y), (255, 0, 0), 2)  # Head line
        
        # Draw center points
        cv2.circle(frame, (shoulder_center_x, shoulder_center_y), 8, (0, 255, 0), -1)  # Shoulder center (green)
        cv2.circle(frame, (hip_center_x, hip_center_y), 8, (0, 255, 0), -1)  # Hip center (green)
        cv2.circle(frame, (head_center_x, head_center_y), 8, (0, 0, 255), -1)  # Head center (red)
        
        # Draw vertical distance lines
        cv2.line(frame, 
                (shoulder_center_x, shoulder_center_y),
                (hip_center_x, hip_center_y),
                (255, 0, 255), 2)  # Torso vertical line (magenta)
        
        cv2.line(frame,
                (head_center_x, head_center_y),
                (shoulder_center_x, shoulder_center_y),
                (255, 0, 255), 2)  # Head to shoulder line (magenta)
        
        # Draw fall threshold line (horizontal line at threshold height)
        threshold_y = int(height * self.alert.fall_threshold)
        cv2.line(frame, 
                (0, threshold_y),
                (width, threshold_y),
                (0, 0, 255), 1)  # Threshold line (red)
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Label shoulder points
        cv2.putText(frame, "LS", (ls_x + 10, ls_y), font, font_scale, (0, 255, 255), thickness)
        cv2.putText(frame, "RS", (rs_x + 10, rs_y), font, font_scale, (0, 255, 255), thickness)
        
        # Label hip points
        cv2.putText(frame, "LH", (lh_x + 10, lh_y), font, font_scale, (255, 255, 0), thickness)
        cv2.putText(frame, "RH", (rh_x + 10, rh_y), font, font_scale, (255, 255, 0), thickness)
        
        # Label head points
        cv2.putText(frame, "N", (nose_x + 10, nose_y), font, font_scale, (255, 0, 0), thickness)
        cv2.putText(frame, "LE", (le_x + 10, le_y), font, font_scale, (255, 0, 0), thickness)
        cv2.putText(frame, "RE", (re_x + 10, re_y), font, font_scale, (255, 0, 0), thickness)
        
        # Label centers
        cv2.putText(frame, "SC", (shoulder_center_x + 10, shoulder_center_y), font, font_scale, (0, 255, 0), thickness)
        cv2.putText(frame, "HC", (hip_center_x + 10, hip_center_y), font, font_scale, (0, 255, 0), thickness)
        cv2.putText(frame, "HD", (head_center_x + 10, head_center_y), font, font_scale, (0, 0, 255), thickness)
        
        # Label threshold
        cv2.putText(frame, f"Fall Threshold: {self.alert.fall_threshold:.2f}", 
                   (10, threshold_y - 10), font, font_scale, (0, 0, 255), thickness)
        
        return frame
        
    def _pose_changed_significantly(self, current_pose) -> bool:
        """Check if the current pose has changed significantly from the last pose.
        
        Args:
            current_pose: Current pose landmarks
            
        Returns:
            True if pose has changed significantly, False otherwise
        """
        if self.last_pose_data is None:
            self.last_pose_data = current_pose
            return True
            
        # Get key points for comparison
        current_points = {
            'shoulder_y': (current_pose.landmark[self.alert.mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
                          current_pose.landmark[self.alert.mp_pose.PoseLandmark.RIGHT_SHOULDER].y) * 0.5,
            'hip_y': (current_pose.landmark[self.alert.mp_pose.PoseLandmark.LEFT_HIP].y + 
                     current_pose.landmark[self.alert.mp_pose.PoseLandmark.RIGHT_HIP].y) * 0.5,
            'nose_y': current_pose.landmark[self.alert.mp_pose.PoseLandmark.NOSE].y
        }
        
        last_points = {
            'shoulder_y': (self.last_pose_data.landmark[self.alert.mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
                          self.last_pose_data.landmark[self.alert.mp_pose.PoseLandmark.RIGHT_SHOULDER].y) * 0.5,
            'hip_y': (self.last_pose_data.landmark[self.alert.mp_pose.PoseLandmark.LEFT_HIP].y + 
                     self.last_pose_data.landmark[self.alert.mp_pose.PoseLandmark.RIGHT_HIP].y) * 0.5,
            'nose_y': self.last_pose_data.landmark[self.alert.mp_pose.PoseLandmark.NOSE].y
        }
        
        # Check if any key point has moved significantly
        threshold = self.thresholds["pose_detection"]["movement_threshold"]
        shoulder_diff = abs(current_points['shoulder_y'] - last_points['shoulder_y'])
        hip_diff = abs(current_points['hip_y'] - last_points['hip_y'])
        nose_diff = abs(current_points['nose_y'] - last_points['nose_y'])
        
        # Only consider it a significant change if multiple points have moved
        significant_movement = (shoulder_diff > threshold and hip_diff > threshold) or \
                             (shoulder_diff > threshold and nose_diff > threshold) or \
                             (hip_diff > threshold and nose_diff > threshold)
        
        if significant_movement:
            self.last_pose_data = current_pose
            return True
            
        return False
        
    def _update_threshold_history(self, category: str, key: str, value: float) -> None:
        """Update the threshold history with new values.
        
        Args:
            category: Category of the threshold
            key: Specific threshold key
            value: New threshold value
        """
        if category == "head_detection":
            if key == "tilt_threshold":
                self.threshold_history["tilt"].append(value)
                if len(self.threshold_history["tilt"]) > self.max_history_length:
                    self.threshold_history["tilt"].pop(0)
            elif key == "position_threshold":
                self.threshold_history["position"].append(value)
                if len(self.threshold_history["position"]) > self.max_history_length:
                    self.threshold_history["position"].pop(0)
            elif key == "shoulder_ratio_threshold":
                self.threshold_history["shoulder_ratio"].append(value)
                if len(self.threshold_history["shoulder_ratio"]) > self.max_history_length:
                    self.threshold_history["shoulder_ratio"].pop(0)
            elif key == "hip_ratio_threshold":
                self.threshold_history["hip_ratio"].append(value)
                if len(self.threshold_history["hip_ratio"]) > self.max_history_length:
                    self.threshold_history["hip_ratio"].pop(0)
                    
    def _log_threshold_history(self) -> None:
        """Log the complete threshold history with colors."""
        logger.info(f"{Fore.CYAN}=== Threshold History ==={Style.RESET_ALL}")
        for metric, history in self.threshold_history.items():
            if history:
                # Calculate statistics
                current = history[-1]
                initial = history[0]
                min_val = min(history)
                max_val = max(history)
                avg = sum(history) / len(history)
                change = ((current - initial) / initial) * 100
                
                # Determine color based on change
                if change > 0:
                    change_color = Fore.RED
                elif change < 0:
                    change_color = Fore.GREEN
                else:
                    change_color = Fore.YELLOW
                
                # Log the statistics
                logger.info(f"{Fore.CYAN}{metric}:{Style.RESET_ALL}")
                logger.info(f"  Current: {Fore.YELLOW}{current:.2f}{Style.RESET_ALL}")
                logger.info(f"  Initial: {Fore.YELLOW}{initial:.2f}{Style.RESET_ALL}")
                logger.info(f"  Min: {Fore.YELLOW}{min_val:.2f}{Style.RESET_ALL}")
                logger.info(f"  Max: {Fore.YELLOW}{max_val:.2f}{Style.RESET_ALL}")
                logger.info(f"  Average: {Fore.YELLOW}{avg:.2f}{Style.RESET_ALL}")
                logger.info(f"  Change: {change_color}{change:+.1f}%{Style.RESET_ALL}")
                logger.info(f"  History: {Fore.YELLOW}{', '.join(f'{x:.2f}' for x in history)}{Style.RESET_ALL}")
                logger.info("")
                
    def _apply_threshold_adjustments(self, adjustments: dict) -> None:
        """Apply threshold adjustments with safety checks."""
        try:
            # Check if auto-update is enabled
            if not self.thresholds["auto_update"]["enabled"]:
                logger.info("Auto-update is disabled, skipping threshold adjustments")
                return
                
            # Check update frequency
            self.update_counter += 1
            logger.debug(f"Update counter: {self.update_counter}/{self.thresholds['auto_update']['update_frequency']}")
            if self.update_counter < self.thresholds["auto_update"]["update_frequency"]:
                logger.debug("Skipping update due to frequency check")
                return
            logger.debug("Update frequency reached, applying adjustments")
            self.update_counter = 0
            
            # Log the current configuration before changes
            logger.info(f"{Fore.CYAN}=== Current Configuration ==={Style.RESET_ALL}")
            for category, values in self.thresholds.items():
                if isinstance(values, dict):
                    logger.info(f"{Fore.YELLOW}{category}:{Style.RESET_ALL}")
                    for key, value in values.items():
                        logger.info(f"  {key}: {Fore.GREEN}{value}{Style.RESET_ALL}")
            
            # Apply adjustments with safety limits
            changes_made = False
            for category, values in adjustments.items():
                if category in self.thresholds:
                    for key, value in values.items():
                        if key in self.thresholds[category]:
                            # Calculate relative change
                            current_value = self.thresholds[category][key]
                            relative_change = abs(value - current_value) / current_value
                            
                            # Apply maximum adjustment limit
                            if relative_change > self.thresholds["auto_update"]["max_adjustment"]:
                                if value > current_value:
                                    value = current_value * (1 + self.thresholds["auto_update"]["max_adjustment"])
                                else:
                                    value = current_value * (1 - self.thresholds["auto_update"]["max_adjustment"])
                                logger.info(f"Limited adjustment for {category}.{key} to {value:.2f}")
                            
                            # Update the threshold and history
                            old_value = self.thresholds[category][key]
                            self.thresholds[category][key] = value
                            self._update_threshold_history(category, key, value)
                            changes_made = True
                            
                            # Log the change
                            change_percent = ((value - old_value) / old_value) * 100
                            change_color = Fore.RED if change_percent > 0 else Fore.GREEN
                            logger.info(f"{Fore.CYAN}Threshold updated:{Style.RESET_ALL}")
                            logger.info(f"  {category}.{key}:")
                            logger.info(f"    Old: {Fore.YELLOW}{old_value:.2f}{Style.RESET_ALL}")
                            logger.info(f"    New: {Fore.YELLOW}{value:.2f}{Style.RESET_ALL}")
                            logger.info(f"    Change: {change_color}{change_percent:+.1f}%{Style.RESET_ALL}")
            
            if changes_made:
                # Save updated thresholds to config file
                config_path = Path(__file__).parent / "config" / "thresholds.json"
                with open(config_path, "w") as f:
                    json.dump(self.thresholds, f, indent=4)
                
                logger.info(f"{Fore.GREEN}✓ Configuration successfully updated and saved{Style.RESET_ALL}")
                self._log_threshold_history()  # Log the complete history after update
            else:
                logger.info(f"{Fore.YELLOW}No threshold adjustments were applied{Style.RESET_ALL}")
            
        except Exception as e:
            logger.error(f"Failed to apply threshold adjustments: {str(e)}")
            logger.error(f"Adjustments attempted: {json.dumps(adjustments, indent=2)}")
            logger.error(f"Current thresholds: {json.dumps(self.thresholds, indent=2)}")
            
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Process a single frame for fall detection and verification."""
        # Reduce frame size by 4x
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (width // 2, height // 2))
        
        # Convert frame to RGB for pose detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe pose detection
        results = self.alert.pose.process(rgb_frame)
        
        # Draw POI if person is detected
        if results.pose_landmarks:
            frame = self.draw_poi(frame, results.pose_landmarks)
            
        # Initialize detection status
        algorithm_detected = False
        llm_detected = False
        
        # Check for potential fall
        if results.pose_landmarks:
            potential_fall = self.alert.detect_fall(results.pose_landmarks)
            algorithm_detected = potential_fall
            threshold_values = None  # Initialize threshold_values
            
            # Additional head-based fall detection with extremely conservative thresholds
            if not potential_fall:
                # Get head landmarks
                nose = results.pose_landmarks.landmark[self.alert.mp_pose.PoseLandmark.NOSE]
                left_ear = results.pose_landmarks.landmark[self.alert.mp_pose.PoseLandmark.LEFT_EAR]
                right_ear = results.pose_landmarks.landmark[self.alert.mp_pose.PoseLandmark.RIGHT_EAR]
                
                # Get shoulder and hip positions for comprehensive checks
                left_shoulder = results.pose_landmarks.landmark[self.alert.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[self.alert.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = results.pose_landmarks.landmark[self.alert.mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = results.pose_landmarks.landmark[self.alert.mp_pose.PoseLandmark.RIGHT_HIP]
                
                # Calculate various position metrics
                head_center_y = (left_ear.y + right_ear.y) * 0.5
                shoulder_center_y = (left_shoulder.y + right_shoulder.y) * 0.5
                hip_center_y = (left_hip.y + right_hip.y) * 0.5
                
                # Calculate ratios with extremely conservative thresholds
                head_to_nose_ratio = abs(nose.y - head_center_y) / abs(left_ear.x - right_ear.x)
                head_to_shoulder_ratio = abs(head_center_y - shoulder_center_y) / abs(left_shoulder.x - right_shoulder.x)
                head_to_hip_ratio = abs(head_center_y - hip_center_y) / abs(left_hip.x - right_hip.x)
                
                # Extremely conservative thresholds for head position
                is_head_tilted = head_to_nose_ratio > self.thresholds["head_detection"]["tilt_threshold"]
                is_head_low = head_center_y > self.alert.fall_threshold + self.thresholds["head_detection"]["position_threshold"]
                is_head_relative_low = head_to_shoulder_ratio > self.thresholds["head_detection"]["shoulder_ratio_threshold"]
                is_head_near_hips = head_to_hip_ratio < self.thresholds["head_detection"]["hip_ratio_threshold"]
                
                # Only consider it a potential fall if ALL conditions are met
                # This makes it much harder to trigger accidentally
                if (is_head_tilted and is_head_low and is_head_relative_low and is_head_near_hips):
                    potential_fall = True
                    algorithm_detected = True
                    threshold_values = {}  # Initialize threshold_values dictionary
                    
                    if is_head_tilted:
                        threshold_values["tilt"] = {
                            "current": head_to_nose_ratio,
                            "threshold": self.thresholds["head_detection"]["tilt_threshold"]
                        }
                    if is_head_low:
                        threshold_values["position"] = {
                            "current": head_center_y,
                            "threshold": self.alert.fall_threshold + self.thresholds["head_detection"]["position_threshold"]
                        }
                    if is_head_relative_low:
                        threshold_values["shoulder_ratio"] = {
                            "current": head_to_shoulder_ratio,
                            "threshold": self.thresholds["head_detection"]["shoulder_ratio_threshold"]
                        }
                    if is_head_near_hips:
                        threshold_values["hip_ratio"] = {
                            "current": head_to_hip_ratio,
                            "threshold": self.thresholds["head_detection"]["hip_ratio_threshold"]
                        }
                    
                    logger.info(f"Potential fall detected based on head position. Triggered thresholds: {', '.join(threshold_values.keys())}")
                else:
                    logger.debug(f"Head position normal: tilt={head_to_nose_ratio:.2f}, low={is_head_low}, relative={head_to_shoulder_ratio:.2f}, hip={head_to_hip_ratio:.2f}")
                
                # Draw threshold information on frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                color = (0, 255, 0)  # Green for normal values
                
                # Draw current values
                y_offset = 20
                # Use red color if threshold is exceeded, green otherwise
                tilt_color = (0, 0, 255) if is_head_tilted else (0, 255, 0)
                cv2.putText(frame, f"Tilt: {head_to_nose_ratio:.2f} / {self.thresholds['head_detection']['tilt_threshold']:.2f}", 
                           (10, y_offset), font, font_scale, tilt_color, thickness)
                y_offset += 20
                
                # Use red color if threshold is exceeded, green otherwise
                position_color = (0, 0, 255) if is_head_low else (0, 255, 0)
                cv2.putText(frame, f"Position: {head_center_y:.2f} / {self.alert.fall_threshold + self.thresholds['head_detection']['position_threshold']:.2f}", 
                           (10, y_offset), font, font_scale, position_color, thickness)
                y_offset += 20
                
                # Use red color if threshold is exceeded, green otherwise
                shoulder_color = (0, 0, 255) if is_head_relative_low else (0, 255, 0)
                cv2.putText(frame, f"Shoulder Ratio: {head_to_shoulder_ratio:.2f} / {self.thresholds['head_detection']['shoulder_ratio_threshold']:.2f}", 
                           (10, y_offset), font, font_scale, shoulder_color, thickness)
                y_offset += 20
                
                # Use red color if threshold is exceeded, green otherwise
                hip_color = (0, 0, 255) if is_head_near_hips else (0, 255, 0)
                cv2.putText(frame, f"Hip Ratio: {head_to_hip_ratio:.2f} / {self.thresholds['head_detection']['hip_ratio_threshold']:.2f}", 
                           (10, y_offset), font, font_scale, hip_color, thickness)
                
                # Add warning text for exceeded thresholds
                y_offset += 20
                if is_head_tilted:
                    cv2.putText(frame, "TILT EXCEEDED", (150, y_offset), font, font_scale, (0, 0, 255), thickness)
                    y_offset += 20
                if is_head_low:
                    cv2.putText(frame, "POSITION EXCEEDED", (150, y_offset), font, font_scale, (0, 0, 255), thickness)
                    y_offset += 20
                if is_head_relative_low:
                    cv2.putText(frame, "SHOULDER RATIO EXCEEDED", (150, y_offset), font, font_scale, (0, 0, 255), thickness)
                    y_offset += 20
                if is_head_near_hips:
                    cv2.putText(frame, "HIP RATIO EXCEEDED", (150, y_offset), font, font_scale, (0, 0, 255), thickness)
                
                # Draw threshold history on the right side
                self._draw_threshold_history()
            
            if potential_fall and not self.fall_confirmed:
                # Check cooldown and pose change
                current_time = time.time()
                if (current_time - self.last_llm_request_time >= self.llm_cooldown and 
                    self._pose_changed_significantly(results.pose_landmarks)):
                    
                    logger.info("Potential fall detected! Waiting for LLM confirmation...")
                    # Save frame for LLM analysis
                    temp_path = "temp_fall_frame.jpg"
                    cv2.imwrite(temp_path, frame)
                    
                    try:
                        # Get LLM analysis with threshold information
                        analysis = self.nanny.analyze_image(temp_path, threshold_values)
                        logger.info(f"LLM Analysis: {analysis}")
                        
                        # Check if LLM confirms the fall
                        llm_detected = "CONFIRMED FALL" in analysis.upper()
                        if llm_detected:
                            logger.info("Fall confirmed by LLM analysis")
                            self.fall_confirmed = True
                            self.current_warning_frames = self.warning_frames
                        else:
                            logger.info("Fall not confirmed by LLM analysis")
                            
                            # Check for threshold adjustment suggestions
                            if "THRESHOLD_ADJUSTMENT:" in analysis:
                                try:
                                    # Extract threshold adjustments from the analysis
                                    adjustment_text = analysis.split("THRESHOLD_ADJUSTMENT:")[1].strip()
                                    
                                    # Handle JSON in markdown code blocks
                                    if "```json" in adjustment_text:
                                        # Extract JSON from code block
                                        json_start = adjustment_text.find("```json") + 7
                                        json_end = adjustment_text.find("```", json_start)
                                        if json_end != -1:
                                            adjustment_text = adjustment_text[json_start:json_end].strip()
                                    else:
                                        # Find the start of the JSON (first '{')
                                        json_start = adjustment_text.find("{")
                                        if json_start != -1:
                                            # Find the matching closing brace
                                            brace_count = 0
                                            for i in range(json_start, len(adjustment_text)):
                                                if adjustment_text[i] == "{":
                                                    brace_count += 1
                                                elif adjustment_text[i] == "}":
                                                    brace_count -= 1
                                                    if brace_count == 0:
                                                        adjustment_text = adjustment_text[json_start:i+1]
                                                        break
                                    
                                    # Remove any newlines and extra spaces
                                    adjustment_text = " ".join(adjustment_text.split())
                                    
                                    # Debug log the extracted JSON
                                    logger.debug(f"Extracted JSON: {adjustment_text}")
                                    
                                    # Parse the JSON
                                    adjustments = json.loads(adjustment_text)
                                    
                                    # Ensure adjustments are properly categorized
                                    if "head_detection" not in adjustments:
                                        # If we have root level thresholds, move them under head_detection
                                        head_detection_adjustments = {}
                                        for key in ["tilt_threshold", "position_threshold", "shoulder_ratio_threshold", "hip_ratio_threshold"]:
                                            if key in adjustments:
                                                head_detection_adjustments[key] = adjustments.pop(key)
                                        if head_detection_adjustments:
                                            adjustments["head_detection"] = head_detection_adjustments
                                    
                                    # Log the parsed adjustments
                                    logger.debug(f"Parsed adjustments: {json.dumps(adjustments, indent=2)}")
                                    
                                    # Apply adjustments with auto-update checks
                                    self._apply_threshold_adjustments(adjustments)
                                    
                                except json.JSONDecodeError as e:
                                    logger.error(f"Failed to parse threshold adjustments JSON: {str(e)}")
                                    logger.debug(f"Raw adjustment text: {adjustment_text}")
                                except Exception as e:
                                    logger.error(f"Failed to process threshold adjustments: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error during LLM analysis: {str(e)}")
                    finally:
                        # Clean up temp file
                        Path(temp_path).unlink(missing_ok=True)
                        # Update last request time
                        self.last_llm_request_time = current_time
                else:
                    logger.debug("Skipping LLM request due to cooldown or unchanged pose")
        
        # Add warning overlay if fall is confirmed
        if self.fall_confirmed and self.current_warning_frames > 0:
            frame = self.alert.add_warning_overlay(frame)
            self.current_warning_frames -= 1
            if self.current_warning_frames == 0:
                logger.info("Fall warning period ended")
                self.fall_confirmed = False
        
        # Update plots after processing frame
        self._draw_threshold_history()
        
        # Update detection history
        self.detection_history["algorithm"].append(1.0 if algorithm_detected else 0.0)
        self.detection_history["llm"].append(1.0 if llm_detected else 0.0)
        self.detection_history["confirmed"].append(1.0 if self.fall_confirmed else 0.0)
        
        # Update error counters
        if algorithm_detected:
            self.error_counters["algorithm"] += 1
        if llm_detected:
            self.error_counters["llm"] += 1
        if self.fall_confirmed:
            self.error_counters["confirmed"] += 1
        
        # Keep history length limited
        for key in self.detection_history:
            if len(self.detection_history[key]) > self.max_history_length:
                self.detection_history[key].pop(0)
        
        # Update detection plot
        x_data = list(range(len(self.detection_history["algorithm"])))
        for detector in ["algorithm", "llm", "confirmed"]:
            self.detection_lines[detector].set_data(x_data, self.detection_history[detector])
        
        # Update error counter text
        error_text = f"Total Detections:\nAlgorithm: {self.error_counters['algorithm']}\nLLM: {self.error_counters['llm']}\nConfirmed: {self.error_counters['confirmed']}"
        self.error_text.set_text(error_text)
        
        # Adjust axes limits
        self.ax_detection.relim()
        self.ax_detection.autoscale_view()
        
        return frame, self.fall_confirmed
        
    def process_video(self, video_path: str) -> None:
        """Process a video file for fall detection.
        
        Args:
            video_path: Path to the video file to process
        """
        logger.info(f"Starting video processing: {video_path}")
        self.is_processing_video = True
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return
            
        try:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames
                    logger.debug(f"Processing frame {frame_count}")
                    
                # Process frame
                processed_frame, fall_detected = self.process_frame(frame)
                
                # Display result
                cv2.imshow("Fall Detection", processed_frame)
                
                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Video processing stopped by user")
                    break
        finally:
            self.is_processing_video = False
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Video processing completed")
        
    def process_image(self, image_path: str) -> Tuple[np.ndarray, bool]:
        """Process a single image for fall detection.
        
        Args:
            image_path: Path to the image file to process
            
        Returns:
            Tuple containing:
            - The processed image with any warnings
            - Boolean indicating if a fall was detected
        """
        self.is_processing_video = False
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        return self.process_frame(frame)
