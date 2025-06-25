import os
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from google.adk.agents import Agent
from google.genai import types
from google import genai
from PIL import Image
from io import BytesIO
from uuid import uuid4
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ImageToImageAgent:
    """Enhanced Image-to-Image Agent that works with direct file paths."""
    
    def __init__(self):
        # Initialize Gemini client for direct image generation
        self.genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Create ADK agent WITHOUT function declarations to avoid the schema error
        self.agent = Agent(
            name="image_to_image_agent",
            model="gemini-2.0-flash-preview-image-generation",
            description="An image-to-image generation assistant that can modify and enhance images using direct file paths.",
            instruction=(
                "You are an image transformation agent. When provided with an image file path and text description, "
                "you can transform, modify, or enhance the image based on the user's request. "
                "You can add objects, change styles, modify scenes, enhance quality, and perform "
                "various other image transformations. You work with direct file paths for efficient processing."
            )
        )
        
    def extract_image_paths_from_text(self, text: str) -> List[str]:
        """Extract image file paths from text with support for multiple formats."""
        # Pattern to match various file path formats
        patterns = [
            # Windows paths with image extensions
            r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*\.(jpg|jpeg|png|gif|bmp|webp|tiff)',
            # Unix paths with image extensions
            r'/(?:[^/\s]+/)*[^/\s]*\.(jpg|jpeg|png|gif|bmp|webp|tiff)',
            # Explicit file path markers from translator
            r'File path: ([^\n\r]+)',
            # URLs with image extensions
            r'https?://[^\s]+\.(jpg|jpeg|png|gif|bmp|webp|tiff)',
        ]
        
        image_paths = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # For patterns with groups, take the first group or the whole match
                    path = match[0] if match[0] else match
                else:
                    path = match
                
                # Clean up the path
                path = path.strip().strip('"\'')
                
                # Check if it's a valid file path (exists) or URL
                if (os.path.exists(path) or path.startswith(('http://', 'https://'))) and path not in image_paths:
                    image_paths.append(path)
        
        logger.info(f"Extracted {len(image_paths)} image paths: {image_paths}")
        return image_paths
    
    def load_image_from_path(self, image_path: str) -> Optional[Image.Image]:
        """Load an image from the given file path."""
        try:
            if os.path.exists(image_path):
                logger.info(f"Loading image from: {image_path}")
                return Image.open(image_path)
            else:
                logger.error(f"Image file not found: {image_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {e}")
            return None
    
    def save_generated_image(self, image_data: bytes, output_path: str = None) -> Optional[str]:
        """Save the generated image to disk."""
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
    
    async def generate_image_to_image_from_path(self, image_path: str, text_prompt: str) -> Tuple[Optional[str], str]:
        """Generate image-to-image transformation using direct file path."""
        try:
            # Load the input image from path
            input_image = self.load_image_from_path(image_path)
            if input_image is None:
                return None, f"Failed to load image from path: {image_path}"
            
            logger.info(f"Processing image: {image_path}")
            logger.info(f"Prompt: {text_prompt}")
            
            # Create the request for Gemini 2.0 Flash Preview image generation
            response = self.genai_client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=[text_prompt, input_image],
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            text_response = ""
            generated_image_path = None
            
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    text_response = part.text
                    logger.info(f"AI Response: {text_response}")
                elif part.inline_data is not None:
                    # Generate output filename based on input
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_path = f"generated_{base_name}_{uuid4().hex[:8]}.png"
                    
                    # Save the generated image
                    generated_image_path = self.save_generated_image(
                        part.inline_data.data,
                        output_path
                    )
            
            success_message = text_response or f"Image transformation completed successfully!"
            if generated_image_path:
                success_message += f"\nGenerated image saved as: {generated_image_path}"
            
            return generated_image_path, success_message
            
        except Exception as e:
            logger.error(f"Error in image generation: {e}")
            return None, f"Error: {str(e)}"
    
    async def process_image_request(self, full_text: str,original_query: str = None) -> Dict[str, Any]:
        """Process a complete image transformation request with path extraction."""
        try:
            # Extract image paths from the text
            image_paths = self.extract_image_paths_from_text(full_text)
            
            if not image_paths:
                return {
                    "status": "error",
                    "message": "No valid image paths found in the request. Please provide the image file path along with your transformation request.\n\nExample: 'Add a cat to this image C:/path/to/image.jpg'"
                }
            
            results = []
            
            # Process each image path
            for image_path in image_paths:
                if original_query and original_query.strip():
                    clean_prompt = original_query.strip()
                    logger.info(f"Using original query as prompt: {clean_prompt}")
                else:
                    
                # Clean the text prompt by removing the image path and file markers
                    clean_prompt = full_text
                    for path in image_paths:
                        clean_prompt = clean_prompt.replace(path, "").strip()
                
                # Remove file path markers from translator
                clean_prompt = re.sub(r'\[FILE PATHS AVAILABLE\].*?(?=\n\n|\Z)', '', clean_prompt, flags=re.DOTALL)
                clean_prompt = re.sub(r'File path: [^\n\r]+', '', clean_prompt)
                clean_prompt = re.sub(r'Note: These are direct file paths.*?(?=\n|\Z)', '', clean_prompt)
                clean_prompt = clean_prompt.strip()
                
                if not clean_prompt:
                    clean_prompt = "Transform this image"
                
                logger.info(f"Processing image: {image_path} with prompt: {clean_prompt}")
                
                # Generate the transformed image
                result_path, ai_response = await self.generate_image_to_image_from_path(image_path, clean_prompt)
                
                results.append({
                    "input_path": image_path,
                    "output_path": result_path,
                    "response": ai_response,
                    "success": result_path is not None
                })
            
            # Create summary response
            if results:
                successful_results = [r for r in results if r["success"]]
                failed_results = [r for r in results if not r["success"]]
                
                summary = f"Image transformation results for {len(image_paths)} image(s):\n\n"
                
                if successful_results:
                    summary += f"✅ Successfully processed {len(successful_results)} image(s):\n"
                    for i, result in enumerate(successful_results, 1):
                        summary += f"{i}. {os.path.basename(result['input_path'])} → {result['output_path']}\n"
                        summary += f"   {result['response']}\n\n"
                
                if failed_results:
                    summary += f"❌ Failed to process {len(failed_results)} image(s):\n"
                    for i, result in enumerate(failed_results, 1):
                        summary += f"{i}. {os.path.basename(result['input_path'])}: {result['response']}\n"
                
                return {
                    "status": "success",
                    "message": summary,
                    "results": results,
                    "successful_count": len(successful_results),
                    "failed_count": len(failed_results)
                }
            else:
                return {
                    "status": "error",
                    "message": "No images were successfully processed."
                }
                
        except Exception as e:
            logger.error(f"Error processing image request: {e}")
            return {
                "status": "error",
                "message": f"Error processing image transformation: {str(e)}"
            }

def extract_image_path_from_text(text: str) -> Optional[str]:
    """Extract the first image path from user text if present."""
    # Look for common image file extensions
    pattern = r'[A-Za-z]:[\\/](?:[^\\/:\*\?"<>\|]+[\\/])*[^\\/:\*\?"<>\|]*\.(jpg|jpeg|png|gif|bmp|webp)'
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    return None

async def main():
    """Main function to handle user input and image processing."""
    print("Enhanced Image-to-Image Generation Agent - Powered by Gemini 2.0 Flash Preview")
    print("=" * 80)
    print("Commands:")
    print("  - Type your image modification request with image path included")
    print("  - Example: 'Add a cat to this image C:/path/to/image.jpg'")
    print("  - Type 'load <path>' to set a default image path")
    print("  - Type 'quit' or 'exit' to exit")
    print("=" * 80)
  
    agent = ImageToImageAgent()
    current_image_path = None
    
    while True:
        try:
            user_input = input("\nEnter your request: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            # Handle image loading command
            if user_input.lower().startswith('load '):
                image_path = user_input[5:].strip()
                # Remove quotes if present
                image_path = image_path.strip('"\'')
                if os.path.exists(image_path):
                    current_image_path = image_path
                    print(f"✅ Image loaded: {current_image_path}")
                else:
                    print(f"❌ Image file not found: {image_path}")
                continue
            
            if not user_input:
                print("Please enter a valid request.")
                continue
            
            # If we have a current image path and the user didn't specify one, add it
            if current_image_path and not agent.extract_image_paths_from_text(user_input):
                user_input = f"{user_input} {current_image_path}"
                print(f"Using current image: {current_image_path}")
            
            print("Processing image transformation...")
            result = await agent.process_image_request(user_input)
            
            print("\n" + "="*50)
            print(result["message"])
            print("="*50)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())