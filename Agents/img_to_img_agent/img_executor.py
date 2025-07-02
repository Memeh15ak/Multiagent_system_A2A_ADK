# mypy: ignore-errors
import logging
import re
import os
import base64
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
import tempfile
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
    
    def extract_image_info_from_request(self, message_parts: List[Any]) -> List[Dict[str, Any]]:
        """Extract image information from the request parts."""
        image_info = []
        
        for part in message_parts:
            # Handle FilePart objects (from A2A framework)
            if hasattr(part, 'root') and isinstance(part.root, FilePart):
                file_part = part.root
                if hasattr(file_part, 'file'):
                    if isinstance(file_part.file, FileWithUri):
                        # File with URI (most common case)
                        uri = file_part.file.uri
                        mime_type = getattr(file_part.file, 'mime_type', 'image/jpeg')
                        
                        # Check if it's an image based on MIME type or file extension
                        if (mime_type and mime_type.startswith('image/')) or \
                           any(uri.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']):
                            image_info.append({
                                'type': 'uri',
                                'uri': uri,
                                'mime_type': mime_type,
                                'is_local_path': os.path.exists(uri)  # Check if it's a local file path
                            })
                            logger.info(f"Found image URI: {uri}")
                    
                    elif isinstance(file_part.file, FileWithBytes):
                        # File with bytes data
                        bytes_data = file_part.file.bytes
                        mime_type = getattr(file_part.file, 'mime_type', 'image/jpeg')
                        
                        if mime_type and mime_type.startswith('image/'):
                            image_info.append({
                                'type': 'bytes',
                                'bytes': bytes_data,
                                'mime_type': mime_type
                            })
                            logger.info(f"Found image bytes data, mime_type: {mime_type}")
            
            # Also check for direct FilePart objects (not wrapped in Part)
            elif isinstance(part, FilePart):
                if hasattr(part, 'file'):
                    if isinstance(part.file, FileWithUri):
                        uri = part.file.uri
                        mime_type = getattr(part.file, 'mime_type', 'image/jpeg')
                        
                        if (mime_type and mime_type.startswith('image/')) or \
                           any(uri.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']):
                            image_info.append({
                                'type': 'uri',
                                'uri': uri,
                                'mime_type': mime_type,
                                'is_local_path': os.path.exists(uri)
                            })
                            logger.info(f"Found direct FilePart image URI: {uri}")
        
        return image_info
    
    def extract_image_paths_from_text(self, text: str) -> List[str]:
        """Extract image file paths from text (fallback method)."""
        # Pattern to match Windows and Unix file paths with image extensions
        patterns = [
            r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*\.(jpg|jpeg|png|gif|bmp|webp|tiff)',  # Windows paths
            r'/(?:[^/\s]+/)*[^/\s]*\.(jpg|jpeg|png|gif|bmp|webp|tiff)',  # Unix paths
            r'Image path: ([^\n\r]+)',  # Explicit image path markers
            r'File path: ([^\n\r]+)',   # Generic file path markers
        ]
        
        image_paths = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # For patterns with groups, take the full match or the path part
                    path = match[0] if len(match) > 1 else match
                else:
                    path = match
                
                # Clean up the path
                path = path.strip().strip('"\'')
                if os.path.exists(path) and path not in image_paths:
                    image_paths.append(path)
        
        return image_paths
    
    def extract_user_prompt_from_text(self, full_text: str, image_paths: List[str]) -> str:
        """Extract the actual user prompt by removing metadata and preserving the core instruction."""
        
        # Start with the original text
        clean_prompt = full_text
        
        # Remove metadata sections (but keep the user's actual instruction)
        metadata_patterns = [
            r'\[IMAGE_FILES\].*?(?=\n[A-Z]|\n[a-z]|$)',  # Remove [IMAGE_FILES] sections
            r'\[FILES\].*?(?=\n[A-Z]|\n[a-z]|$)',        # Remove [FILES] sections
            r'\[IMAGE PATHS AVAILABLE\].*?(?=\n[A-Z]|\n[a-z]|$)',
            r'\[FILE PATHS AVAILABLE\].*?(?=\n[A-Z]|\n[a-z]|$)',
            r'Image path: [^\n\r]+',                      # Remove individual path lines
            r'File path: [^\n\r]+',
            r'Path: [^\n\r]+',
            r'File URL: [^\n\r]+',
            r'Note: These are direct file paths.*?(?=\n|\Z)',
            r'Please process the above image.*?(?=\n|\Z)',
            r'\d+\.\s*File (path|URL): [^\n\r]+',
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
    
    def load_image_from_bytes(self, image_bytes: bytes) -> Optional[Image.Image]:
        """Load an image from bytes data."""
        try:
            # Handle base64 encoded bytes
            if isinstance(image_bytes, str):
                image_bytes = base64.b64decode(image_bytes)
                
            return Image.open(BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Error loading image from bytes: {e}")
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
    """An AgentExecutor that runs an ADK-based Image-to-Image Agent with improved file handling."""

    def __init__(self):
        # Don't initialize the agent here - do it lazily when needed
        self._agent = None
        self.runner = None

    async def _get_agent_and_runner(self):
        """Lazily initialize the agent and runner when first needed."""
        if self._agent is None:
            # Create the agent instance
            self._agent = ImageToImageAgent()
            self.runner = Runner(
                app_name=self._agent.agent.name,
                agent=self._agent.agent,
                artifact_service=InMemoryArtifactService(),
                session_service=InMemorySessionService(),
                memory_service=InMemoryMemoryService(),
            )
        return self._agent, self.runner

    def _run_agent(
        self,
        session_id: str,
        new_message: genai_types.Content,
        task_updater: TaskUpdater,
    ) -> AsyncGenerator[Event, None]:
        """Runs the ADK agent with the given message."""
        return self.runner.run_async(
            session_id=session_id,
            user_id='self',
            new_message=new_message,
        )

    async def _process_request(
        self,
        new_message: genai_types.Content,
        session_id: str,
        task_updater: TaskUpdater,
        original_message_parts: List[Any],
    ) -> AsyncIterable[TaskStatus | Artifact]:
        """Processes the incoming request with improved image handling."""
        # Ensure agent and runner are initialized
        agent, runner = await self._get_agent_and_runner()
        
        # Extract text parts from the message
        text_parts = []
        for part in new_message.parts:
            if part.text:
                text_parts.append(part.text)
        
        if not text_parts:
            error_response = [TextPart(text="No text content found in the request. Please provide both image and transformation description.")]
            task_updater.add_artifact(error_response)
            task_updater.complete()
            return
        
        # Combine all text parts
        full_text = " ".join(text_parts)
        logger.info(f"Processing request with text: {full_text[:200]}...")
        
        # PRIORITY 1: Extract image info from the original A2A message parts
        image_info_list = agent.extract_image_info_from_request(original_message_parts)
        logger.info(f"Extracted {len(image_info_list)} images from message parts")
        
        # PRIORITY 2: Fallback to text extraction if no images found in parts
        if not image_info_list:
            image_paths = agent.extract_image_paths_from_text(full_text)
            logger.info(f"Extracted image paths from text: {image_paths}")
            
            # Convert paths to image info format
            for path in image_paths:
                image_info_list.append({
                    'type': 'uri',
                    'uri': path,
                    'mime_type': 'image/jpeg',
                    'is_local_path': True
                })
        
        if not image_info_list:
            error_response = [TextPart(text="No images found in the request. Please provide an image file along with your transformation request.")]
            task_updater.add_artifact(error_response)
            task_updater.complete()
            return
        
        try:
            # Update status to indicate processing
            task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message([
                    TextPart(text=f"Processing image transformation for {len(image_info_list)} image(s)...")
                ])
            )
            
            results = []
            
            # Process each image
            for i, image_info in enumerate(image_info_list):
                # Extract the clean user prompt using the improved method
                all_image_paths = [info['uri'] for info in image_info_list if info['type'] == 'uri']
                clean_prompt = agent.extract_user_prompt_from_text(full_text, all_image_paths)
                
                logger.info(f"Clean prompt for image {i+1}: {clean_prompt}")
                
                # Load the image based on type
                input_image = None
                image_source = ""
                
                if image_info['type'] == 'uri':
                    if image_info.get('is_local_path', False):
                        input_image = agent.load_image_from_path(image_info['uri'])
                        image_source = os.path.basename(image_info['uri'])
                    else:
                        # Handle remote URI - would need additional implementation
                        logger.warning(f"Remote URI not yet supported: {image_info['uri']}")
                        continue
                
                elif image_info['type'] == 'bytes':
                    input_image = agent.load_image_from_bytes(image_info['bytes'])
                    image_source = f"bytes_data_{i+1}"
                
                if input_image is None:
                    error_text = f"❌ Failed to load image {i+1}: {image_source}"
                    results.append({'text': error_text})
                    continue
                
                # Generate the transformed image
                result_path, ai_response = await agent.generate_image_to_image(input_image, clean_prompt, image_source)
                
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
                summary_text = f"Image transformation results for {len(image_info_list)} image(s):\n\n"
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

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        """Executes the agent's logic based on the incoming A2A request."""
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            updater.submit()
        updater.start_work()
        await self._process_request(
            genai_types.UserContent(
                parts=convert_a2a_parts_to_genai(context.message.parts),
            ),
            context.context_id,
            updater,
            context.message.parts,  # Pass original A2A parts for image extraction
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
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


def convert_genai_parts_to_a2a(parts: list[genai_types.Part]) -> list[Part]:
    """Converts a list of Google GenAI Part objects to a list of A2A Part objects."""
    return [
        convert_genai_part_to_a2a(part)
        for part in parts
        if (part.text or part.file_data or part.inline_data)
    ]


def convert_genai_part_to_a2a(part: genai_types.Part) -> Part:
    """Converts a single Google GenAI Part object to an A2A Part object."""
    if part.text:
        return TextPart(text=part.text)
    if part.file_data:
        return FilePart(
            file=FileWithUri(
                uri=part.file_data.file_uri,
                mime_type=part.file_data.mime_type,
            )
        )
    if part.inline_data:
        # Convert bytes to base64 string for FileWithBytes
        data_base64 = base64.b64encode(part.inline_data.data).decode('utf-8')
        return Part(
            root=FilePart(
                file=FileWithBytes(
                    bytes=data_base64,  # Store as base64 string
                    mime_type=part.inline_data.mime_type,
                )
            )
        )
    raise ValueError(f'Unsupported part type: {part}')