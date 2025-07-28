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

class ImageGenerationAgent:
    """Simplified Image Generation Agent using Gemini API."""
    
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def save_generated_image(self, image_data: bytes, output_path: str = None) -> Optional[str]:
        """Save generated image to disk."""
        try:
            # Set output directory inside container
            output_dir = "/MAS/generated_images"
            os.makedirs(output_dir, exist_ok=True)

            # If no specific output_path given, generate one
            if not output_path:
                output_path = f"generated_image_{uuid4().hex[:8]}.png"
            
            # Always save inside the mounted directory
            full_output_path = os.path.join(output_dir, os.path.basename(output_path))

            image = Image.open(BytesIO(image_data))
            image.save(full_output_path)
            logger.info(f"✅ Generated image saved as: {full_output_path}")
            return full_output_path

        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None

    def enhance_prompt(self, prompt: str) -> str:
        """
        Enhances the user prompt with additional context for better image generation.
        """
        if len(prompt) > 300:  # Don't enhance already detailed prompts
            return prompt
        
        # Check if quality terms are already present
        quality_terms = ["high quality", "detailed", "professional", "sharp", "masterpiece", "4k", "8k"]
        has_quality = any(term in prompt.lower() for term in quality_terms)
        
        if not has_quality:
            # Add quality modifiers
            if "portrait" in prompt.lower():
                prompt = f"{prompt}, high quality portrait, professional lighting, sharp focus"
            elif "landscape" in prompt.lower():
                prompt = f"{prompt}, stunning landscape, high resolution, vibrant colors"
            elif "art" in prompt.lower() or "painting" in prompt.lower():
                prompt = f"{prompt}, masterpiece, detailed artwork, professional quality"
            else:
                prompt = f"{prompt}, high quality, detailed, professional"
        
        return prompt
    
    async def generate_image(self, prompt: str) -> Tuple[Optional[str], str]:
        """Generate image using Gemini API."""
        try:
            # Enhance the prompt
            enhanced_prompt = self.enhance_prompt(prompt)
            logger.info(f"Processing enhanced prompt: {enhanced_prompt}")
            
            # Generate content with the enhanced prompt
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=[enhanced_prompt],
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE'],
                    temperature=0.8  # Slightly higher for more creativity
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
                    output_path = f"generated_{uuid4().hex[:8]}.png"
                    
                    # Save generated image
                    generated_image_path = self.save_generated_image(
                        part.inline_data.data, output_path
                    )
            
            success_message = text_response or "Image generation completed!"
            if generated_image_path:
                success_message += f"\nSaved as: {generated_image_path}"
            
            return generated_image_path, success_message
            
        except Exception as e:
            logger.error(f"Error in image generation: {e}")
            return None, f"Error: {str(e)}"

async def main():
    """Main function for interactive image generation."""
    print("Image Generation Agent - Gemini API")
    print("=" * 50)
    print("Usage: Enter a prompt to generate an image")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    agent = ImageGenerationAgent()
    
    while True:
        try:
            # Get generation prompt
            prompt = input("\nGeneration prompt: ").strip()
            if prompt.lower() in ['quit', 'exit']:
                break
            
            if not prompt:
                print("❌ Please enter a generation prompt!")
                continue
            
            print("🔄 Processing...")
            result_path, message = await agent.generate_image(prompt)
            
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