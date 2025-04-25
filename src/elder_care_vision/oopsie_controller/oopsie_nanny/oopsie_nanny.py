"""The sensible filter that prevents false alarms! 🧐

This module implements the OopsieNanny - the rational verifier that prevents
overly-sensitive fall detection from causing unnecessary panic. It uses OpenAI's
vision capabilities to provide a second opinion on potential falls.
"""

import base64
import os
import logging
from pathlib import Path
from typing import Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO to reduce logging
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ImageRecognizer:
    """A class that uses OpenAI's vision capabilities to analyze images.
    
    This class provides methods to:
    1. Encode images for API transmission
    2. Analyze images for fall detection
    3. Process API responses
    """
    
    def __init__(self):
        """Initialize the ImageRecognizer with API configuration."""
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=self.api_key)
        logger.info("ImageRecognizer initialized with OpenAI API key")
        
    def encode_image(self, image_path: str) -> str:
        """Encode an image file to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        try:
            with open(image_path, "rb") as image_file:
                # Read the image file
                image_data = image_file.read()
                # Verify the image data is not empty
                if not image_data:
                    raise ValueError(f"Empty image file: {image_path}")
                # Encode to base64
                encoded_string = base64.b64encode(image_data).decode("utf-8")
                # Verify the encoded string is not empty
                if not encoded_string:
                    raise ValueError(f"Failed to encode image: {image_path}")
                logger.debug(f"Successfully encoded image: {image_path}")
                return encoded_string
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            raise
            
    def analyze_image(self, image_path: str, threshold_values: Optional[dict] = None) -> str:
        """Analyze an image for potential falls using OpenAI's API.
        
        Args:
            image_path: Path to the image file to analyze
            threshold_values: Dictionary containing threshold information that triggered the detection
            
        Returns:
            Analysis result from the API
        """
        # Encode the image
        base64_image = self.encode_image(image_path)
        
        # Prepare the prompt with threshold information
        threshold_info = ""
        if threshold_values:
            threshold_info = "\n\nThreshold Information:\n"
            for metric, values in threshold_values.items():
                threshold_info += f"- {metric}: Current value {values['current']:.2f} exceeded threshold {values['threshold']:.2f}\n"
        
        prompt = """Analyze this image for potential fall detection. Consider:
        1. Body position and orientation
        2. Head position relative to shoulders
        3. Overall posture and stability
        4. Environmental context
        5. Signs of distress or discomfort

        Provide a detailed analysis of at least 100 words, considering all these factors.
        Conclude with a clear verdict: either 'CONFIRMED FALL' or 'NO FALL'.

        If you believe the thresholds need adjustment, provide a THRESHOLD_ADJUSTMENT section with suggested new values in JSON format.
        The thresholds should be under the 'head_detection' category and can include:
        - tilt_threshold (current: 2.0)
        - position_threshold (current: 0.36)
        - shoulder_ratio_threshold (current: 2.07)
        - hip_ratio_threshold (current: 1.35)

        Example format:
        THRESHOLD_ADJUSTMENT:
        {
            "head_detection": {
                "tilt_threshold": 2.1,
                "position_threshold": 0.38,
                "shoulder_ratio_threshold": 2.1,
                "hip_ratio_threshold": 1.4
            }
        }"""
        
        logger.info("Sending request to OpenAI API")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            logger.info("Received response from OpenAI API")
            analysis = response.choices[0].message.content
            logger.info(f"Analysis result: {analysis}")
            
            # Check if this is a fall
            is_fall = "CONFIRMED FALL" in analysis.upper()
            logger.info(f"Fall confirmation status: {is_fall}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error during API call: {str(e)}")
            raise

def main():
    # Example usage
    recognizer = ImageRecognizer()
    
    # Get the path to the image
    image_path = "image.png"
    
    try:
        # Analyze the image
        result = recognizer.analyze_image(image_path)
        print("\nAnalysis Result:")
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 