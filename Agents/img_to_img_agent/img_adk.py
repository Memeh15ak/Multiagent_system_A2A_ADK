import os
import logging
from typing import Optional, Tuple
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from uuid import uuid4
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ImageToImageAgent:
    """Simplified Image-to-Image Agent using Gemini Files API."""
    
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def upload_image(self, image_path: str) -> Optional[str]:
        """Upload image to Gemini Files API and return file URI."""
        try:
            # Upload file to Gemini
            uploaded_file = self.client.files.upload(path=image_path)
            logger.info(f"Uploaded image: {uploaded_file.uri}")
            return uploaded_file.uri
        except Exception as e:
            logger.error(f"Error uploading image: {e}")
            return None
    
    def save_generated_image(self, image_data: bytes, output_path: str = None) -> Optional[str]:
        """Save generated image to disk."""
        try:
            if not output_path:
                output_path = f"generated_image_{uuid4().hex[:8]}.png"
            
            image = Image.open(BytesIO(image_data))
            image.save(output_path)
            logger.info(f"Generated image saved as: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None
    
    async def transform_image(self, image_path: str, prompt: str) -> Tuple[Optional[str], str]:
        """Transform image using Gemini Files API."""
        try:
            # Upload image to Gemini Files API
            file_uri = self.upload_image(image_path)
            if not file_uri:
                return None, "Failed to upload image to Gemini Files API"
            
            logger.info(f"Processing image with prompt: {prompt}")
            
            # Generate content with uploaded file
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=[
                    prompt,
                    types.Part(file_data=types.FileData(file_uri=file_uri))
                ],
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            text_response = ""
            generated_image_path = None
            
            # Process response parts
            for part in response.candidates[0].content.parts:
                if part.text:
                    text_response = part.text
                    logger.info(f"AI Response: {text_response}")
                elif part.inline_data:
                    # Generate output filename
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_path = f"generated_{base_name}_{uuid4().hex[:8]}.png"
                    
                    # Save generated image
                    generated_image_path = self.save_generated_image(
                        part.inline_data.data, output_path
                    )
            
            success_message = text_response or "Image transformation completed!"
            if generated_image_path:
                success_message += f"\nSaved as: {generated_image_path}"
            
            return generated_image_path, success_message
            
        except Exception as e:
            logger.error(f"Error in image transformation: {e}")
            return None, f"Error: {str(e)}"

async def main():
    """Main function for interactive image processing."""
    print("Image-to-Image Agent - Gemini Files API")
    print("=" * 50)
    print("Usage: Enter image path and transformation prompt")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    agent = ImageToImageAgent()
    
    while True:
        try:
            # Get image path
            image_path = input("\nImage path: ").strip().strip('"\'')
            if image_path.lower() in ['quit', 'exit']:
                break
            
            if not os.path.exists(image_path):
                print("❌ Image file not found!")
                continue
            
            # Get transformation prompt
            prompt = input("Transformation prompt: ").strip()
            if not prompt:
                print("❌ Please enter a transformation prompt!")
                continue
            
            print("🔄 Processing...")
            result_path, message = await agent.transform_image(image_path, prompt)
            
            print("\n" + "="*50)
            if result_path:
                print("✅ SUCCESS:")
            else:
                print("❌ FAILED:")
            print(message)
            print("="*50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())