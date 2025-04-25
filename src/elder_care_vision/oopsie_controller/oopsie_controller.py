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
from concurrent.futures import ThreadPoolExecutor

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
        frame_history: List to store last 6 frames
        frame_timestamps: List to store timestamps for each frame
        max_history_frames: Number of frames to keep in history
        frame_interval: Seconds between frames (6 frames in 3 seconds)
        last_frame_time: Track last frame time
        frame_queue: Queue for incoming frames
        analysis_queue: Queue for analysis results
        thread_pool: ThreadPoolExecutor for analysis tasks
        is_analyzing: Boolean to track analysis status
        queue_status: Dictionary to store queue status
        algorithm_fall_subscribers: List of subscribers for algorithm-detected falls
        confirmed_fall_subscribers: List of subscribers for LLM-confirmed falls
    """
    
    def __init__(self):
        """Initialize the OopsieController with its components."""
        self.alert = FallDetector()
        self.nanny = ImageRecognizer()
        
        # Initialize subscribers
        self.algorithm_fall_subscribers = []
        self.confirmed_fall_subscribers = []
        
        # Initialize frame history
        self.frame_history = []  # Store last 6 frames
        self.frame_timestamps = []  # Store timestamps for each frame
        self.max_history_frames = 6  # Number of frames to keep in history
        self.frame_interval = 0.5  # Seconds between frames (6 frames in 3 seconds)
        self.last_frame_time = 0  # Track last frame time
        
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
        
        # Initialize matplotlib figures with smaller sizes (30% smaller)
        self.fig_thresholds, self.axs_thresholds = plt.subplots(2, 2, figsize=(5.6, 4.2))  # 30% smaller
        self.fig_detection, self.ax_detection = plt.subplots(figsize=(5.6, 2.1))  # 30% smaller
        
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
        
        # Initialize queues and thread pool
        self.frame_queue = queue.Queue(maxsize=30)  # Max 30 frames in queue
        self.analysis_queue = queue.Queue()  # Queue for analysis results
        self.thread_pool = ThreadPoolExecutor(max_workers=2)  # 2 worker threads for analysis
        self.is_analyzing = False
        self.queue_status = {
            'total_queued': 0,
            'total_processed': 0,
            'dropped_frames': 0
        }

        # Start analysis worker thread
        self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.analysis_thread.start()
        
        logger.info("OopsieController initialized and ready for fall detection")
        
    def _run_plot_window(self) -> None:
        """Run the tkinter window in a separate thread."""
        try:
            # Create tkinter window
            self.root = tk.Tk()
            self.root.title("Fall Detection Monitor")
            
            # Calculate window size based on plot dimensions
            window_width = 1120  # 560 * 2 for plots and sequence
            window_height = 800  # Increased height to accommodate log widget
            
            # Set window size and make it non-resizable
            self.root.geometry(f"{window_width}x{window_height}")
            self.root.resizable(False, False)
            self.root.configure(bg='#1e1e1e')
            
            # Create main frame
            main_frame = tk.Frame(self.root, bg='#1e1e1e')
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create left frame for plots
            left_frame = tk.Frame(main_frame, bg='#1e1e1e', width=560)
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Create right frame for sequence and logs
            right_frame = tk.Frame(main_frame, bg='#1e1e1e', width=560)
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            # Create plot frames in left frame
            threshold_frame = tk.Frame(left_frame, bg='#1e1e1e', height=420)
            threshold_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
            
            detection_frame = tk.Frame(left_frame, bg='#1e1e1e', height=210)
            detection_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
            
            # Create sequence frame in right frame
            sequence_frame = tk.Frame(right_frame, bg='#1e1e1e')
            sequence_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create log frame below sequence
            log_frame = tk.Frame(right_frame, bg='#1e1e1e')
            log_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create labels for plot images
            self.threshold_label = tk.Label(threshold_frame, bg='#1e1e1e')
            self.threshold_label.pack(fill=tk.BOTH, expand=True)
            
            self.detection_label = tk.Label(detection_frame, bg='#1e1e1e')
            self.detection_label.pack(fill=tk.BOTH, expand=True)
            
            # Create label for sequence image
            self.sequence_label = tk.Label(sequence_frame, bg='#1e1e1e')
            self.sequence_label.pack(fill=tk.BOTH, expand=True)
            
            # Create queue status label
            self.queue_status_label = tk.Label(
                log_frame,
                text="Queue Status: Initializing...",
                bg='#1e1e1e',
                fg='white',
                font=('Courier', 10),
                justify=tk.LEFT
            )
            self.queue_status_label.pack(fill=tk.X, padx=10, pady=5)
            
            # Create scrolled text widget for logs
            self.log_text = scrolledtext.ScrolledText(
                log_frame,
                bg='#2d2d2d',
                fg='white',
                font=('Courier', 10),
                height=10
            )
            self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Start periodic updates
            self._update_plots()
            self._update_queue_status()
            self._update_logs()
            
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
            
            # Update sequence image if available and has changed
            if hasattr(self, 'last_sequence_path') and Path(self.last_sequence_path).exists():
                try:
                    # Check if file has been modified since last update
                    current_mtime = Path(self.last_sequence_path).stat().st_mtime
                    if not hasattr(self, 'last_sequence_mtime') or current_mtime > self.last_sequence_mtime:
                        sequence_image = Image.open(self.last_sequence_path)
                        
                        # Calculate target size maintaining aspect ratio
                        target_width = 560  # Half of window width
                        target_height = 630  # Total height of plots
                        
                        # Get original dimensions
                        orig_width, orig_height = sequence_image.size
                        
                        # Calculate aspect ratios
                        target_ratio = target_width / target_height
                        orig_ratio = orig_width / orig_height
                        
                        # Calculate new dimensions maintaining aspect ratio
                        if orig_ratio > target_ratio:
                            # Image is wider than target ratio
                            new_width = target_width
                            new_height = int(target_width / orig_ratio)
                        else:
                            # Image is taller than target ratio
                            new_height = target_height
                            new_width = int(target_height * orig_ratio)
                        
                        # Resize image maintaining aspect ratio
                        sequence_image = sequence_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        # Create a new image with black background
                        final_image = Image.new('RGB', (target_width, target_height), (30, 30, 30))
                        
                        # Calculate position to center the image
                        x = (target_width - new_width) // 2
                        y = (target_height - new_height) // 2
                        
                        # Paste the resized image onto the center of the black background
                        final_image.paste(sequence_image, (x, y))
                        
                        # Convert to PhotoImage
                        sequence_photo = ImageTk.PhotoImage(final_image)
                        self.sequence_label.configure(image=sequence_photo)
                        self.sequence_label.image = sequence_photo
                        
                        # Update last modification time
                        self.last_sequence_mtime = current_mtime
                except Exception as e:
                    logger.error(f"Error updating sequence image: {str(e)}")
            
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
                
                # Update text with more detailed stats
                current = history[-1]
                initial = history[0]
                avg = sum(history) / len(history)
                change = ((current - initial) / initial) * 100
                
                stats_text = (
                    f"Current: {current:.2f}\n"
                    f"Initial: {initial:.2f}\n"
                    f"Avg: {avg:.2f}\n"
                    f"Change: {change:+.1f}%"
                )
                self.text_boxes[metric].set_text(stats_text)
                
                # Adjust axes limits with padding
                ax = self.plot_lines[metric].axes
                y_min = min(history) * 0.95  # Add 5% padding
                y_max = max(history) * 1.05
                ax.set_ylim(y_min, y_max)
                ax.set_xlim(-1, len(history))
            
            # Force redraw of the figure
            self.fig_thresholds.canvas.draw()
            self.fig_thresholds.canvas.flush_events()
            
        except Exception as e:
            logger.error(f"Error updating threshold history plots: {str(e)}")
            
    def _update_threshold_history(self, category: str, key: str, value: float) -> None:
        """Update the threshold history with new values.
        
        Args:
            category: Category of the threshold
            key: Specific threshold key
            value: New threshold value
        """
        try:
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
                        
            # Force an immediate redraw of the threshold plots
            self._draw_threshold_history()
            
        except Exception as e:
            logger.error(f"Error updating threshold history: {str(e)}")
            logger.error(f"Category: {category}, Key: {key}, Value: {value}")
            
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
            
    def _create_frame_sequence(self) -> Optional[str]:
        """Create a sequence image from frame history with timestamps.
        
        Returns:
            Path to the saved sequence image, or None if not enough frames
        """
        if len(self.frame_history) < self.max_history_frames:
            return None
            
        try:
            # Calculate dimensions for the sequence image
            frame_height, frame_width = self.frame_history[0].shape[:2]
            sequence_width = frame_width * 3  # 3 frames per row
            sequence_height = frame_height * 2  # 2 rows
            
            # Create blank image for the sequence
            sequence = np.zeros((sequence_height, sequence_width, 3), dtype=np.uint8)
            
            # Add frames to sequence with timestamps
            for i, (frame, timestamp) in enumerate(zip(self.frame_history, self.frame_timestamps)):
                row = i // 3
                col = i % 3
                y_start = row * frame_height
                x_start = col * frame_width
                
                # Add frame to sequence
                sequence[y_start:y_start + frame_height, x_start:x_start + frame_width] = frame
                
                # Add timestamp with milliseconds
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                time_diff = self.frame_timestamps[-1] - timestamp
                timestamp_str = f"t-{time_diff:.3f}s"
                cv2.putText(sequence, timestamp_str, 
                           (x_start + 10, y_start + frame_height - 10),
                           font, font_scale, (255, 255, 255), thickness)
            
            # Save sequence image with unique filename
            sequence_path = f"temp_frame_sequence_{int(time.time()*1000)}.jpg"
            cv2.imwrite(sequence_path, sequence)
            
            # Clean up old sequence file if it exists
            if hasattr(self, 'last_sequence_path') and Path(self.last_sequence_path).exists():
                try:
                    Path(self.last_sequence_path).unlink()
                except Exception as e:
                    logger.error(f"Error cleaning up old sequence file: {str(e)}")
            
            self.last_sequence_path = sequence_path  # Store path for display
            return sequence_path
            
        except Exception as e:
            logger.error(f"Error creating frame sequence: {str(e)}")
            return None

    def _analysis_worker(self):
        """Worker thread that processes frames from the queue."""
        while True:
            try:
                # Get frame data from queue
                frame_data = self.frame_queue.get()
                if frame_data is None:  # Sentinel value to stop the thread
                    break

                frame, landmarks, timestamp = frame_data
                self.queue_status['total_processed'] += 1

                # Check for potential fall
                if landmarks is not None:
                    potential_fall = self.alert.detect_fall(landmarks)
                    if potential_fall and not self.fall_confirmed:
                        current_time = time.time()
                        if (current_time - self.last_llm_request_time >= self.llm_cooldown and 
                            self._pose_changed_significantly(landmarks)):
                            
                            logger.info("Potential fall detected! Queueing for LLM analysis...")
                            
                            # Create and save frame sequence
                            sequence_path = self._create_frame_sequence()
                            if sequence_path:
                                # Submit analysis task to thread pool
                                self.thread_pool.submit(self._analyze_fall, sequence_path, landmarks)

                self.frame_queue.task_done()

            except Exception as e:
                logger.error(f"Error in analysis worker: {str(e)}")

    def _analyze_fall(self, sequence_path: str, landmarks):
        """Analyze potential fall in a separate thread."""
        try:
            # Calculate threshold values
            threshold_values = self._calculate_threshold_values(landmarks)
            
            # Get LLM analysis
            analysis = self.nanny.analyze_image(sequence_path, threshold_values)
            logger.info(f"LLM Analysis: {analysis}")
            
            # Put result in analysis queue
            self.analysis_queue.put((analysis, threshold_values))
            
        except Exception as e:
            logger.error(f"Error during fall analysis: {str(e)}")
        finally:
            # Clean up temp file
            Path(sequence_path).unlink(missing_ok=True)
            self.last_llm_request_time = time.time()

    def _calculate_threshold_values(self, landmarks):
        """Calculate threshold values for fall detection."""
        # ... (implement threshold calculations)
        return {}

    def _update_logs(self) -> None:
        """Update the log text widget with new messages from the queue."""
        try:
            while True:
                try:
                    # Get message from queue without blocking
                    message = log_queue.get_nowait()
                    
                    # Enable text widget for editing
                    self.log_text.configure(state='normal')
                    
                    # Add message with timestamp
                    self.log_text.insert(tk.END, f"{message}\n")
                    
                    # Scroll to bottom
                    self.log_text.see(tk.END)
                    
                    # Disable text widget for editing
                    self.log_text.configure(state='disabled')
                    
                except queue.Empty:
                    break
                    
            # Schedule next update
            if hasattr(self, 'root'):
                self.root.after(100, self._update_logs)
                
        except Exception as e:
            logger.error(f"Error updating logs: {str(e)}")
            
    def _update_queue_status(self) -> None:
        """Update queue status display in tkinter window."""
        try:
            if hasattr(self, 'queue_status_label'):
                status_text = (
                    f"Queue Status:\n"
                    f"Frames Queued: {self.frame_queue.qsize()}\n"
                    f"Total Queued: {self.queue_status['total_queued']}\n"
                    f"Total Processed: {self.queue_status['total_processed']}\n"
                    f"Dropped Frames: {self.queue_status['dropped_frames']}"
                )
                self.queue_status_label.config(text=status_text)
                
            # Schedule next update
            if hasattr(self, 'root'):
                self.root.after(100, self._update_queue_status)
                
        except Exception as e:
            logger.error(f"Error updating queue status: {str(e)}")

    def add_algorithm_fall_subscriber(self, callback: callable) -> None:
        """Add a subscriber for algorithm-detected falls.
        
        Args:
            callback: Function to call when algorithm detects a fall.
                     Will be called with (frame, landmarks, timestamp)
        """
        self.algorithm_fall_subscribers.append(callback)
        
    def add_confirmed_fall_subscriber(self, callback: callable) -> None:
        """Add a subscriber for LLM-confirmed falls.
        
        Args:
            callback: Function to call when LLM confirms a fall.
                     Will be called with (frame_history, frame_timestamps, analysis_text, timestamp)
        """
        self.confirmed_fall_subscribers.append(callback)
        
    def _notify_algorithm_fall(self, frame: np.ndarray, landmarks, timestamp: float) -> None:
        """Notify all algorithm fall subscribers.
        
        Args:
            frame: The frame where fall was detected
            landmarks: The pose landmarks
            timestamp: Time of detection
        """
        for subscriber in self.algorithm_fall_subscribers:
            try:
                subscriber(frame.copy(), landmarks, timestamp)
            except Exception as e:
                logger.error(f"Error in algorithm fall subscriber: {str(e)}")
                
    def _notify_confirmed_fall(self, sequence_path: str, analysis: str, timestamp: float) -> None:
        """Notify all confirmed fall subscribers.
        
        Args:
            sequence_path: Path to the frame sequence image (unused)
            analysis: The LLM analysis text
            timestamp: Time of confirmation
        """
        for subscriber in self.confirmed_fall_subscribers:
            try:
                # Pass the actual frame history and timestamps instead of the sequence path
                subscriber(self.frame_history.copy(), self.frame_timestamps.copy(), analysis, timestamp)
            except Exception as e:
                logger.error(f"Error in confirmed fall subscriber: {str(e)}")
                
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Process a single frame for fall detection."""
        # Reduce frame size by 4x
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (width // 2, height // 2))
        
        # Convert frame to RGB for pose detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe pose detection
        results = self.alert.pose.process(rgb_frame)
        
        # Update frame history
        current_time = time.time()
        
        # Only update frame history every 0.5 seconds
        if not self.frame_timestamps or (current_time - self.frame_timestamps[-1] >= 0.5):
            self.frame_history.append(frame.copy())
            self.frame_timestamps.append(current_time)
            
            # Keep only the last N frames
            if len(self.frame_history) > self.max_history_frames:
                self.frame_history.pop(0)
                self.frame_timestamps.pop(0)
        
        # Draw POI if person is detected
        if results.pose_landmarks:
            frame = self.draw_poi(frame, results.pose_landmarks)
            
            # Try to add frame to queue, drop if full
            try:
                frame_data = (frame.copy(), results.pose_landmarks, current_time)
                self.frame_queue.put_nowait(frame_data)
                self.queue_status['total_queued'] += 1
                logger.debug(f"Frame queued. Queue size: {self.frame_queue.qsize()}")
                
                # Check for potential fall
                if self.alert.detect_fall(results.pose_landmarks):
                    current_time = time.time()
                    if (current_time - self.last_llm_request_time >= self.llm_cooldown and 
                        self._pose_changed_significantly(results.pose_landmarks)):
                        
                        # Notify algorithm fall subscribers at this point
                        self._notify_algorithm_fall(frame, results.pose_landmarks, current_time)
                        
                        logger.info("Potential fall detected! Queueing for LLM analysis...")
                        
                        # Create and save frame sequence
                        sequence_path = self._create_frame_sequence()
                        if sequence_path:
                            # Submit analysis task to thread pool
                            self.thread_pool.submit(self._analyze_fall, sequence_path, results.pose_landmarks)
                    
            except queue.Full:
                self.queue_status['dropped_frames'] += 1
                logger.warning("Frame queue full, dropping frame")

        # Check for analysis results
        try:
            while not self.analysis_queue.empty():
                analysis, threshold_values = self.analysis_queue.get_nowait()
                
                # Process analysis result
                if "CONFIRMED FALL" in analysis.upper():
                    logger.info("Fall confirmed by LLM analysis")
                    self.fall_confirmed = True
                    self.current_warning_frames = self.warning_frames
                    
                    # Notify confirmed fall subscribers
                    if hasattr(self, 'frame_history') and hasattr(self, 'frame_timestamps'):
                        self._notify_confirmed_fall(
                            None,
                            analysis,
                            time.time()
                        )
                
                # Check for threshold adjustments
                if "THRESHOLD_ADJUSTMENT:" in analysis:
                    self._process_threshold_adjustment(analysis)

        except queue.Empty:
            pass

        # Add warning overlay if fall is confirmed
        if self.fall_confirmed and self.current_warning_frames > 0:
            frame = self.alert.add_warning_overlay(frame)
            self.current_warning_frames -= 1
            if self.current_warning_frames == 0:
                logger.info("Fall warning period ended")
                self.fall_confirmed = False

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
                    logger.info(f"Queue Status - Size: {self.frame_queue.qsize()}, " +
                              f"Processed: {self.queue_status['total_processed']}, " +
                              f"Dropped: {self.queue_status['dropped_frames']}")
                    
                # Process frame
                processed_frame, fall_detected = self.process_frame(frame)
                
                # Display result
                cv2.imshow("Fall Detection", processed_frame)
                
                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Video processing stopped by user")
                    break
                    
                # Small delay to allow queue processing
                time.sleep(0.001)
                
        finally:
            self.is_processing_video = False
            cap.release()
            cv2.destroyAllWindows()
            
            # Wait for remaining frames to be processed
            logger.info("Waiting for remaining frames to be processed...")
            self.frame_queue.join()
            logger.info("Video processing completed")
            
    def __del__(self):
        """Cleanup when the controller is destroyed."""
        # Stop analysis worker
        if hasattr(self, 'frame_queue'):
            self.frame_queue.put(None)
        
        # Shutdown thread pool
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)

    def _process_threshold_adjustment(self, analysis: str) -> None:
        """Process threshold adjustments from LLM analysis.
        
        Args:
            analysis: The analysis string containing threshold adjustments
        """
        try:
            # Extract the JSON part between THRESHOLD_ADJUSTMENT: and the next newline
            start_idx = analysis.find("THRESHOLD_ADJUSTMENT:") + len("THRESHOLD_ADJUSTMENT:")
            json_str = analysis[start_idx:].strip()
            
            # Parse the JSON adjustments
            adjustments = json.loads(json_str)
            logger.info(f"Processing threshold adjustments: {json.dumps(adjustments, indent=2)}")
            
            # Apply the adjustments
            self._apply_threshold_adjustments(adjustments)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse threshold adjustments: {str(e)}")
            logger.error(f"Raw adjustment string: {json_str}")
        except Exception as e:
            logger.error(f"Error processing threshold adjustments: {str(e)}")
            logger.error(f"Analysis text: {analysis}")

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
