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
    """Enhanced image-to-image agent that handles direct file paths."""
    
    def __init__(self):
        # Initialize Gemini client for direct image generation
        self.genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Create ADK agent without tools to avoid schema validation issues
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
        """Extract image file paths from text."""
        # Pattern to match Windows and Unix file paths with image extensions
        patterns = [
            r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*\.(jpg|jpeg|png|gif|bmp|webp|tiff)',  # Windows paths
            r'/(?:[^/\s]+/)*[^/\s]*\.(jpg|jpeg|png|gif|bmp|webp|tiff)',  # Unix paths
            r'File path: ([^\n\r]+)',  # Explicit file path markers from translator
        ]
        
        image_paths = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # For patterns with groups, take the full match
                    path = match[0] if len(match) > 1 else match
                else:
                    path = match
                
                # Clean up the path
                path = path.strip().strip('"\'')
                if os.path.exists(path) and path not in image_paths:
                    image_paths.append(path)
        
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
    
    async def generate_image_to_image_from_path(self, image_path: str, text_prompt: str) -> tuple[Optional[str], str]:
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

class ADKImageToImageAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs an ADK-based Image-to-Image Agent with direct path support."""

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
    ) -> AsyncIterable[TaskStatus | Artifact]:
        """Processes the incoming request by running the ADK agent with direct path support."""
        # Ensure agent and runner are initialized
        agent, runner = await self._get_agent_and_runner()
        
        # Extract text parts from the message
        text_parts = []
        for part in new_message.parts:
            if part.text:
                text_parts.append(part.text)
        
        if not text_parts:
            error_response = [TextPart(text="No text content found in the request. Please provide both image path and transformation description.")]
            task_updater.add_artifact(error_response)
            task_updater.complete()
            return
        
        # Combine all text parts
        full_text = " ".join(text_parts)
        logger.info(f"Processing request with text: {full_text[:200]}...")
        
        # Extract image paths from the text
        image_paths = agent.extract_image_paths_from_text(full_text)
        logger.info(f"Extracted image paths: {image_paths}")
        
        if not image_paths:
            error_response = [TextPart(text="No valid image paths found in the request. Please provide the image file path along with your transformation request.\n\nExample: 'Add a cat to this image C:/path/to/image.jpg'")]
            task_updater.add_artifact(error_response)
            task_updater.complete()
            return
        
        try:
            # Update status to indicate processing
            task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message([
                    TextPart(text=f"Processing image transformation for {len(image_paths)} image(s)...")
                ])
            )
            
            results = []
            
            # Process each image path
            for image_path in image_paths:
                # Clean the text prompt by removing the image path
                clean_prompt = full_text
                for path in image_paths:
                    clean_prompt = clean_prompt.replace(path, "").strip()
                
                # Remove file path markers from translator
                clean_prompt = re.sub(r'\[IMAGE FILES TO PROCESS\].*?(?=Please process|\Z)', '', clean_prompt, flags=re.DOTALL)
                clean_prompt = re.sub(r'\[FILE PATHS AVAILABLE\].*?(?=Note:|$)', '', clean_prompt, flags=re.DOTALL)
                clean_prompt = re.sub(r'File path: [^\n\r]+', '', clean_prompt)
                clean_prompt = re.sub(r'File URL: [^\n\r]+', '', clean_prompt)
                clean_prompt = re.sub(r'Note: These are direct file paths.*?(?=\n|\Z)', '', clean_prompt)
                clean_prompt = re.sub(r'Please process the above image.*?(?=\n|\Z)', '', clean_prompt)
                clean_prompt = re.sub(r'\d+\.\s*File (path|URL): [^\n\r]+', '', clean_prompt)
                clean_prompt = ' '.join(clean_prompt.split())
                clean_prompt = clean_prompt.strip()
                
                if not clean_prompt:
                    logger.info("clened prompt do not exist")
                    break
                
                logger.info(f"Clean prompt for {image_path}: {clean_prompt}")
                
                # Generate the transformed image
                result_path, ai_response = await agent.generate_image_to_image_from_path(image_path, clean_prompt)
                
                if result_path and os.path.exists(result_path):
                    # Read the generated image
                    with open(result_path, 'rb') as f:
                        image_data = f.read()
                    
                    # Create response with the generated image
                    response_text = f"✅ Image transformation completed for: {os.path.basename(image_path)}\n{ai_response}"
                    
                    results.append({
                        'text': response_text,
                        'image_data': image_data,
                        'image_path': result_path
                    })
                else:
                    # Error generating image
                    error_text = f"❌ Failed to transform image: {os.path.basename(image_path)}\n{ai_response}"
                    results.append({'text': error_text})
            
            # Create final response
            if results:
                response_parts = []
                
                # Add text summary
                summary_text = f"Image transformation results for {len(image_paths)} image(s):\n\n"
                for i, result in enumerate(results, 1):
                    summary_text += f"{i}. {result['text']}\n"
                
                response_parts.append(TextPart(text=summary_text))
                
                # Add generated images - FIX: Convert bytes to base64 string
                for result in results:
                    if 'image_data' in result:
                        # Convert bytes to base64 string for FileWithBytes
                        image_base64 = base64.b64encode(result['image_data']).decode('utf-8')
                        response_parts.append(FilePart(file=FileWithBytes(
                            bytes=image_base64,  # Now passing base64 string instead of raw bytes
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
