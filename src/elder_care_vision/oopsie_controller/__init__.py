"""The grand conductor of your fall detection drama! 🎭

This is the oopsie-controller - the mastermind behind your fall detection ecosystem.
It orchestrates the overly-sensitive detector (oopsie_alert) and the sensible filter (oopsie_nanny)
to create a balanced and effective fall detection system.

Components:
    - OopsieController: The main conductor, integrating all components
    - FallDetector: The alarm-happy detector that triggers on every movement
    - ImageRecognizer: The rational filter that checks "Is this really a fall?"

Why "oopsie"? Because sometimes life happens, and we're here to catch those moments! 🎪
"""

from .oopsie_controller import OopsieController
from .oopsie_alert import FallDetector
from .oopsie_nanny import ImageRecognizer

__all__ = ["OopsieController", "FallDetector", "ImageRecognizer"] 