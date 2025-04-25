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
            
    def analyze_image(self, image_path: str) -> str:
        """Analyze an image using OpenAI's vision capabilities.
        
        Args:
            image_path: Path to the image file to analyze
            
        Returns:
            Analysis result as a string
        """
        try:
            # Verify the image file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            # Encode the image
            base64_image = self.encode_image(image_path)
            
            # Prepare the API request
            prompt = (
                "Analyze this image for potential falls with a detailed 100-word analysis. "
                "Consider the following aspects:\n"
                "1. Body position and orientation\n"
                "2. Head position relative to body\n"
                "3. Overall posture and balance\n"
                "4. Environmental context\n"
                "5. Any signs of distress or instability\n\n"
                "Provide a comprehensive analysis of at least 100 words explaining why this "
                "situation may or may not constitute a fall. Include specific observations "
                "about body angles, positions, and any contextual factors that support your "
                "conclusion. End your analysis with a clear verdict: 'CONFIRMED FALL' if a "
                "fall is definitely occurring, or 'NO FALL' if the situation appears normal."
            )
            
            logger.info("Sending request to OpenAI API")
            logger.debug(f"Request details: Image path={image_path}")
            
            # Make the API call using the new client interface
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",  # Changed back to the correct model
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
            
            # Log the API response summary
            logger.info("Received response from OpenAI API")
            logger.debug(f"API response: Model={response.model}, Tokens used={response.usage.total_tokens}")
            
            # Extract and log the analysis result
            analysis = response.choices[0].message.content.strip()
            logger.info(f"Analysis result: {analysis}")
            
            # Check if the analysis confirms a fall
            is_fall_confirmed = "CONFIRMED FALL" in analysis.upper()
            logger.info(f"Fall confirmation status: {is_fall_confirmed}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
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