# mypy: ignore-errors
import logging
import re
import os
import base64
import tempfile
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Dict, Any, Optional, List
from uuid import uuid4
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.events import Event
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Artifact,
    FilePart,
    FileWithBytes,
    FileWithUri,
    Part,
    TaskState,
    TaskStatus,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import get_text_parts
from a2a.utils.errors import ServerError
from google.adk.agents import Agent
from google import genai
from PIL import Image
from io import BytesIO


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ImageToImageAgent:
    """Enhanced image-to-image agent that handles both file paths and FilePart objects."""
    
    def __init__(self):
        # Initialize Gemini client for direct image generation
        self.genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Create ADK agent without tools to avoid schema validation issues
        self.agent = Agent(
            name="image_to_image_agent",
            model="gemini-2.0-flash-preview-image-generation",
            description="An image-to-image generation assistant that can modify and enhance images.",
            instruction=(
                "You are an image transformation agent. When provided with an image and text description, "
                "you can transform, modify, or enhance the image based on the user's request. "
                "You can add objects, change styles, modify scenes, enhance quality, and perform "
                "various other image transformations."
            )
        )
    
    def extract_image_paths_from_text(self, text: str) -> List[str]:
        """Extract image file paths from text with improved regex patterns."""
        # Improved patterns to handle various path formats
        patterns = [
            # Windows paths with backslashes: C:\folder\image.jpg
            r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*\.(jpg|jpeg|png|gif|bmp|webp|tiff)',
            
            # Windows paths with forward slashes: C:/folder/image.jpg
            r'[A-Za-z]:/(?:[^/\s:*?"<>|\r\n]+/)*[^/\s:*?"<>|\r\n]*\.(jpg|jpeg|png|gif|bmp|webp|tiff)',
            
            # Unix absolute paths: /path/to/image.jpg
            r'/(?:[^/\s]+/)*[^/\s]*\.(jpg|jpeg|png|gif|bmp|webp|tiff)',
            
            # Relative paths: ./image.jpg or ../folder/image.jpg or just image.jpg
            r'(?:\.{1,2}/)?(?:[^/\s]+/)*[^/\s]*\.(jpg|jpeg|png|gif|bmp|webp|tiff)',
            
            # Explicit markers
            r'Image path:\s*([^\n\r]+)',
            r'File path:\s*([^\n\r]+)',
            r'Path:\s*([^\n\r]+)',
            
            # Generic file path patterns (more flexible)
            r'([A-Za-z]:[/\\](?:[^/\\:*?"<>|\r\n\s]+[/\\])*[^/\\:*?"<>|\r\n\s]*\.(jpg|jpeg|png|gif|bmp|webp|tiff))',
        ]
        
        image_paths = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # For patterns with groups, find the actual path
                    path = None
                    for group in match:
                        if group and ('.' in group or '/' in group or '\\' in group):
                            path = group
                            break
                    if not path:
                        path = match[0]  # Fallback to first group
                else:
                    path = match
                
                # Clean up the path
                path = path.strip().strip('"\'').strip()
                
                # Validate the path
                if path and len(path) > 3:  # Minimum meaningful path length
                    # Check if it's a valid image file extension
                    if any(path.lower().endswith(f'.{ext}') for ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff']):
                        # Check if file exists
                        if os.path.exists(path):
                            if path not in image_paths:
                                image_paths.append(path)
                                logger.info(f"Found valid image path: {path}")
                        else:
                            # Even if file doesn't exist, log the path for debugging
                            logger.warning(f"Image path found but file doesn't exist: {path}")
        
        return image_paths
    
    def extract_user_prompt_from_text(self, full_text: str, image_paths: List[str]) -> str:
        """Extract the actual user prompt by removing metadata and preserving the core instruction."""
        
        # Start with the original text
        clean_prompt = full_text
        
        # Remove metadata sections (but keep the user's actual instruction)
        metadata_patterns = [
            r'The user has provided image files located at.*?(?=\n\n|\n[A-Z]|\Z)',
            r'User request:\s*',
            r'Image path:\s*[^\n\r]+',
            r'File path:\s*[^\n\r]+',
            r'Path:\s*[^\n\r]+',
        ]
        
        for pattern in metadata_patterns:
            clean_prompt = re.sub(pattern, '', clean_prompt, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove actual file paths from the text
        for path in image_paths:
            clean_prompt = clean_prompt.replace(path, '').strip()
        
        # Clean up multiple whitespaces and newlines
        clean_prompt = re.sub(r'\s+', ' ', clean_prompt).strip()
        
        # If the prompt is too short or empty, extract it differently
        if len(clean_prompt) < 5:
            # Try to find the actual user instruction by looking for common command words
            instruction_patterns = [
                r'(modify|change|add|remove|transform|enhance|create|make|generate|turn|convert).*',
                r'"([^"]*)"',  # Text in quotes
                r"'([^']*)'",  # Text in single quotes
            ]
            
            for pattern in instruction_patterns:
                matches = re.findall(pattern, full_text, re.IGNORECASE | re.DOTALL)
                if matches:
                    # Take the first meaningful match
                    match = matches[0]
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else match[1]
                    if len(match.strip()) > 10:  # Minimum meaningful length
                        clean_prompt = match.strip()
                        break
        
        # Final fallback
        if len(clean_prompt) < 5:
            clean_prompt = "Transform this image"
        
        return clean_prompt
    
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
    
    async def generate_image_to_image(self, input_image: Image.Image, text_prompt: str, image_source: str = "input") -> tuple[Optional[str], str]:
        """Generate image-to-image transformation using PIL Image object."""
        try:
            logger.info(f"Processing image from: {image_source}")
            logger.info(f"Prompt: {text_prompt}")
            
            # Create the request for Gemini 2.0 Flash Preview image generation
            response = self.genai_client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=[text_prompt, input_image],
                config=genai_types.GenerateContentConfig(
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
                    # Generate output filename
                    output_path = f"generated_{uuid4().hex[:8]}.png"
                    
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

class ADKImageToImageAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs an ADK-based Image-to-Image Agent following the working pattern."""

    def __init__(self):
        # Don't initialize the agent here - do it lazily when needed
        self._agent = None
        self.runner = None
        self.image_agent = ImageToImageAgent()

    async def _get_agent_and_runner(self):
        """Lazily initialize the agent and runner when first needed."""
        if self._agent is None:
            # Create the agent instance
            self._agent = self.image_agent.agent
            self.runner = Runner(
                app_name=self._agent.name,
                agent=self._agent,
                artifact_service=InMemoryArtifactService(),
                session_service=InMemorySessionService(),
                memory_service=InMemoryMemoryService(),
            )
        return self._agent, self.runner

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        """Executes the agent's logic based on the incoming A2A request."""
        logger.info(f"Starting execution for task {context.task_id}")
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        updater.submit()
        updater.start_work()

        temp_files_to_clean = []
        try:
            # Prepare message and files following the working pattern
            augmented_prompt, temp_files_to_clean = await self._prepare_message_and_files(context.message.parts)
            
            # Create genai content
            genai_content = genai_types.UserContent(parts=[genai_types.Part(text=augmented_prompt)])
            
            # Ensure agent and runner are initialized
            await self._get_agent_and_runner()

            # Process the request
            await self._process_request(genai_content, context.context_id, updater, temp_files_to_clean)

        except Exception as e:
            logger.error(f"Error in execute for task {context.task_id}: {e}", exc_info=True)
            updater.failed(error_message=f"Execution failed: {str(e)}")
        finally:
            # Clean up temporary files
            for path in temp_files_to_clean:
                try:
                    os.remove(path)
                    logger.info(f"Cleaned up temporary file: {path}")
                except OSError as e:
                    logger.error(f"Error removing temporary file {path}: {e}")

    async def _prepare_message_and_files(self, parts: list[Part]) -> tuple[str, list[str]]:
        """Saves file parts to temp files and creates an augmented prompt for the ADK agent."""
        temp_file_paths = []
        text_parts = []
        
        for part in parts:
            part_root = part.root
            if isinstance(part_root, FilePart) and isinstance(part_root.file, FileWithBytes):
                file_suffix = os.path.splitext(part_root.file.filename or '.tmp')[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
                    tmp_file.write(part_root.file.bytes)
                    temp_file_paths.append(tmp_file.name)
                    logger.info(f"Saved incoming file to temporary path: {tmp_file.name}")
            elif isinstance(part_root, TextPart):
                text_parts.append(part_root.text)

        user_prompt = " ".join(text_parts)
        
        # Also extract image paths from the text (for cases where paths are mentioned in text)
        image_paths_from_text = self.image_agent.extract_image_paths_from_text(user_prompt)
        if image_paths_from_text:
            logger.info(f"Found image paths in text: {image_paths_from_text}")
            temp_file_paths.extend(image_paths_from_text)
        
        if temp_file_paths:
            path_info = "The user has provided image files located at the following paths: " + \
                        ", ".join(f"'{p}'" for p in temp_file_paths) + ". "
            return f"{path_info}\n\nUser request: {user_prompt}", temp_file_paths
        
        return user_prompt, temp_file_paths

    async def _process_request(
        self,
        new_message: genai_types.Content,
        session_id: str,
        task_updater: TaskUpdater,
        temp_file_paths: list[str],
    ):
        """Process the image transformation request."""
        try:
            # Extract text from the message
            text_parts = []
            for part in new_message.parts:
                if part.text:
                    text_parts.append(part.text)
            
            full_text = " ".join(text_parts)
            logger.info(f"Processing request with text: {full_text[:200]}...")
            
            if not temp_file_paths:
                error_response = [TextPart(text="No images found in the request. Please provide an image file along with your transformation request.")]
                task_updater.add_artifact(error_response)
                task_updater.complete()
                return
            
            # Update status to indicate processing
            task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message([
                    TextPart(text=f"Processing image transformation for {len(temp_file_paths)} image(s)...")
                ])
            )
            
            results = []
            
            # Process each image
            for i, image_path in enumerate(temp_file_paths):
                # Extract the clean user prompt
                clean_prompt = self.image_agent.extract_user_prompt_from_text(full_text, temp_file_paths)
                
                logger.info(f"Clean prompt for image {i+1}: {clean_prompt}")
                
                # Load the image
                input_image = self.image_agent.load_image_from_path(image_path)
                image_source = os.path.basename(image_path)
                
                if input_image is None:
                    error_text = f"❌ Failed to load image {i+1}: {image_source} (path: {image_path})"
                    logger.error(error_text)
                    results.append({'text': error_text})
                    continue
                
                logger.info(f"Successfully loaded image: {image_source}")
                
                # Generate the transformed image
                result_path, ai_response = await self.image_agent.generate_image_to_image(input_image, clean_prompt, image_source)
                
                if result_path and os.path.exists(result_path):
                    # Read the generated image
                    with open(result_path, 'rb') as f:
                        image_data = f.read()
                    
                    # Create response with the generated image
                    response_text = f"✅ Image transformation completed for: {image_source}\nPrompt used: {clean_prompt}\n{ai_response}"
                    
                    results.append({
                        'text': response_text,
                        'image_data': image_data,
                        'image_path': result_path
                    })
                else:
                    # Error generating image
                    error_text = f"❌ Failed to transform image: {image_source}\n{ai_response}"
                    results.append({'text': error_text})
            
            # Create final response
            if results:
                response_parts = []
                
                # Add text summary
                summary_text = f"Image transformation results for {len(temp_file_paths)} image(s):\n\n"
                for i, result in enumerate(results, 1):
                    summary_text += f"{i}. {result['text']}\n"
                
                response_parts.append(TextPart(text=summary_text))
                
                # Add generated images
                for result in results:
                    if 'image_data' in result:
                        # Convert bytes to base64 string for FileWithBytes
                        image_base64 = base64.b64encode(result['image_data']).decode('utf-8')
                        response_parts.append(FilePart(file=FileWithBytes(
                            bytes=image_base64,
                            mime_type="image/png"
                        )))
                
                task_updater.add_artifact(response_parts)
                task_updater.complete()
            else:
                error_response = [TextPart(text="No images were successfully processed.")]
                task_updater.add_artifact(error_response)
                task_updater.complete()
                
        except Exception as e:
            logger.error(f"Error processing image-to-image request: {e}")
            error_response = [TextPart(text=f"Error processing image transformation: {str(e)}")]
            task_updater.add_artifact(error_response)
            task_updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        logger.warning(f"Cancellation requested for task {context.task_id}, but not supported.")
        raise ServerError(error=UnsupportedOperationError())

    async def _upsert_session(self, session_id: str):
        """Retrieves or creates an ADK session."""
        return await self.runner.session_service.get_session(
            app_name=self.runner.app_name, user_id='self', session_id=session_id
        ) or await self.runner.session_service.create_session(
            app_name=self.runner.app_name, user_id='self', session_id=session_id
        )


def convert_a2a_parts_to_genai(parts: list[Part]) -> list[genai_types.Part]:
    """Converts a list of A2A Part objects to a list of Google GenAI Part objects."""
    return [convert_a2a_part_to_genai(part) for part in parts]


def convert_a2a_part_to_genai(part: Part) -> genai_types.Part:
    """Converts a single A2A Part object to a Google GenAI Part object."""
    part = part.root
    if isinstance(part, TextPart):
        return genai_types.Part(text=part.text)
    if isinstance(part, FilePart):
        if isinstance(part.file, FileWithUri):
            return genai_types.Part(
                file_data=genai_types.FileData(
                    file_uri=part.file.uri, mime_type=part.file.mime_type
                )
            )
        if isinstance(part.file, FileWithBytes):
            # Handle both base64 string and raw bytes
            if isinstance(part.file.bytes, str):
                # Already base64 encoded
                data = base64.b64decode(part.file.bytes)
            else:
                # Raw bytes
                data = part.file.bytes
            
            return genai_types.Part(
                inline_data=genai_types.Blob(
                    data=data, mime_type=part.file.mime_type
                )
            )
        raise ValueError(f'Unsupported file type: {type(part.file)}')
    raise ValueError(f'Unsupported part type: {type(part)}')


def convert_genai_parts_to_a2a(parts: list[genai_types.Part] | None) -> list[Part]:
    """Converts a list of Google GenAI Part objects to a list of A2A Part objects."""
    if not parts:
        return []
    return [Part(root=TextPart(text=p.text)) for p in parts if p.text]