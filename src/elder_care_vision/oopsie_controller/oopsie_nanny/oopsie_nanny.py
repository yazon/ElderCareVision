import os
from pathlib import Path
import base64
from openai import OpenAI
from dotenv import load_dotenv

class ImageRecognizer:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.client = OpenAI(api_key=api_key)
        
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def analyze_image(self, image_path: str, prompt: str | None = None) -> str:
        """
        Analyze an image using OpenAI's vision model.
        
        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt for the analysis
            
        Returns:
            str: Analysis result from the model
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Default prompt if none provided
        if prompt is None:
            prompt = """Analyze for fall risk: Check person's position, environment hazards, and movement. Rate risk: Low/Medium/High."""
        
        # Encode the image
        base64_image = self.encode_image(image_path)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using vision model for image analysis
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
                max_tokens=250,  # Limiting response length
                temperature=0.3  # Lower temperature for more focused responses
            )
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error analyzing image: {str(e)}")

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