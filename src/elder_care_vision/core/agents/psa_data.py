from dataclasses import dataclass


@dataclass
class FallDetectionResult:
    confidence_level: int = -1
    """Represents the confidence of a fall between 0 and 100."""
    fall_image: str = ""
    """Represents the base64 encoded image of a fall."""
    analysis: str = ""
    """Represents the analysis of a fall."""
