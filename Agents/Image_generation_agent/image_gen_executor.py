import asyncio
import logging
import os
import json
import base64
from pathlib import Path
from typing import Optional, Tuple, List, Any, Union
from uuid import uuid4
from collections.abc import AsyncGenerator, AsyncIterable

from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.events import Event
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from google import genai
from PIL import Image
from io import BytesIO

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    FilePart, FileWithBytes, FileWithUri, Part, TaskState, TextPart,
    UnsupportedOperationError, TaskStatus, Artifact
)
from a2a.utils import get_text_parts
from a2a.utils.errors import ServerError
from google.adk.agents import Agent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ImageGenerationAgent:
    """Enhanced Image Generation Agent using Gemini API with improved structure."""
    
    def __init__(self):
        # Initialize Gemini client for direct image generation
        self.genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Create ADK agent without tools to avoid schema validation issues
        self.agent = Agent(
            name="image_generation_agent",
            model="gemini-2.0-flash-exp-image-generation",
            description="An AI image generation assistant that creates images from text descriptions.",
            instruction=(
                "You are an image generation agent. When provided with a text description, "
                "you create high-quality images that match the user's request. "
                "You can generate images in various styles, compositions, and subjects "
                "based on detailed text prompts."
            )
        )
    
    def extract_text_from_message(self, message):
        """
        Extract text from various message formats more reliably.
        """
        try:
            logger.info(f"🔍 Extracting text from message type: {type(message)}")
            
            # Method 1: Handle string directly
            if isinstance(message, str):
                logger.info(f"📝 Direct string message: {message[:50]}...")
                return message.strip()
            
            # Method 2: Handle list of message parts
            if isinstance(message, list):
                text_parts = []
                for item in message:
                    logger.info(f"🔍 Processing list item type: {type(item)}")
                    
                    # Handle Part objects with text
                    if hasattr(item, 'text') and item.text:
                        text_parts.append(item.text.strip())
                        logger.info(f"📝 Found text: {item.text[:50]}...")
                    
                    # Handle objects with root.text structure
                    elif hasattr(item, 'root') and hasattr(item.root, 'text') and item.root.text:
                        text_parts.append(item.root.text.strip())
                        logger.info(f"📝 Found root text: {item.root.text[:50]}...")
                    
                    # Handle nested parts
                    elif hasattr(item, 'parts'):
                        for part in item.parts:
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text.strip())
                                logger.info(f"📝 Found nested text: {part.text[:50]}...")
                    
                    # Handle plain string
                    elif isinstance(item, str):
                        text_parts.append(item.strip())
                        logger.info(f"📝 Found string: {item[:50]}...")
                
                if text_parts:
                    result = " ".join(text_parts).strip()
                    logger.info(f"✅ Extracted from list: {result[:100]}...")
                    return result
            
            # Method 3: Handle single object with text
            if hasattr(message, 'text') and message.text:
                logger.info(f"📝 Found direct text: {message.text[:50]}...")
                return message.text.strip()
            
            # Method 4: Handle object with parts
            if hasattr(message, 'parts') and message.parts:
                text_parts = []
                for part in message.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text.strip())
                        logger.info(f"📝 Found part text: {part.text[:50]}...")
                
                if text_parts:
                    result = " ".join(text_parts).strip()
                    logger.info(f"✅ Extracted from parts: {result[:100]}...")
                    return result
            
            # Method 5: Try get_text_parts utility
            try:
                text_parts = get_text_parts(message)
                if text_parts:
                    result = " ".join(text_parts).strip()
                    logger.info(f"✅ Extracted using utility: {result[:100]}...")
                    return result
            except Exception as e:
                logger.warning(f"⚠️ get_text_parts failed: {e}")
            
            # Method 6: Last resort - string conversion
            if hasattr(message, '__str__'):
                result = str(message).strip()
                if result and result != str(type(message)):
                    logger.info(f"🔄 Fallback string: {result[:100]}...")
                    return result
            
            logger.warning(f"⚠️ No text found in message")
            logger.warning(f"Message type: {type(message)}")
            logger.warning(f"Message attributes: {[attr for attr in dir(message) if not attr.startswith('_')]}")
            
            return ""
            
        except Exception as e:
            logger.error(f"❌ Error extracting text: {e}", exc_info=True)
            return ""

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

    async def generate_image(self, prompt: str) -> Tuple[Optional[str], str]:
        """Generate image using Gemini API."""
        try:
            # Enhance the prompt
            enhanced_prompt = self.enhance_prompt(prompt)
            logger.info(f"Processing enhanced prompt: {enhanced_prompt}")
            
            # Generate content with the enhanced prompt
            response = self.genai_client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=[enhanced_prompt],
                config=genai_types.GenerateContentConfig(
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

class ADKImageGeneratorExecutor(AgentExecutor):
    """An optimized AgentExecutor for Gemini 2.0 Flash image generation."""

    def __init__(self):
        # Don't initialize the agent here - do it lazily when needed
        self._agent = None
        self.runner = None

    async def _get_agent_and_runner(self):
        """Lazily initialize the agent and runner when first needed."""
        if self._agent is None:
            # Create the agent instance
            self._agent = ImageGenerationAgent()
            self.runner = Runner(
                app_name=self._agent.agent.name,
                agent=self._agent.agent,
                artifact_service=InMemoryArtifactService(),
                session_service=InMemorySessionService(),
                memory_service=InMemoryMemoryService(),
            )
            logger.info("🔧 Initialized ADK Image Generator Executor")
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
            user_id='a2a_user',
            new_message=new_message,
        )

    async def _process_request(
        self,
        new_message: genai_types.Content,
        session_id: str,
        task_updater: TaskUpdater,
        original_message_parts: List[Any],
    ) -> AsyncIterable[TaskStatus | Artifact]:
        """Processes the incoming request with improved handling."""
        # Ensure agent and runner are initialized
        agent, runner = await self._get_agent_and_runner()
        
        # Extract text from the message
        prompt = agent.extract_text_from_message(original_message_parts)
        
        logger.info(f"📝 Extracted prompt: '{prompt}'")
        
        # Enhanced validation
        if not prompt:
            logger.error("❌ No prompt extracted from message")
            error_response = [Part(root=TextPart(text="No text content found in the message. Please provide a detailed description for the image you want to generate."))]
            task_updater.add_artifact(error_response)
            task_updater.complete()
            return
        
        if len(prompt.strip()) < 3:
            logger.error(f"❌ Prompt too short: '{prompt}'")
            error_response = [Part(root=TextPart(text="Please provide a more detailed description for the image you want to generate."))]
            task_updater.add_artifact(error_response)
            task_updater.complete()
            return
        
        # Clean up the prompt
        prompt = prompt.strip()
        
        # Remove common system prefixes if present
        prefixes_to_remove = [
            "Generate image based on:",
            "Create image:",
            "Make image:",
            "Draw:",
            "Paint:",
            "Design:"
        ]
        
        for prefix in prefixes_to_remove:
            if prompt.lower().startswith(prefix.lower()):
                prompt = prompt[len(prefix):].strip()
                logger.info(f"🔧 Cleaned prompt: '{prompt}'")
                break
        
        logger.info(f"✅ Final prompt: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
        
        try:
            # Update status to indicate processing
            task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message([
                    Part(root=TextPart(text=f"Generating image based on: {prompt[:100]}..."))
                ])
            )
            
            # Generate the image
            generated_image_path, ai_response = await agent.generate_image(prompt)
            
            artifacts = []
            
            # Add text response if available
            if ai_response:
                artifacts.append(Part(root=TextPart(text=ai_response)))
                logger.info(f"📝 Added text response: {ai_response[:100]}...")
            
            # Add generated image if available
            if generated_image_path and os.path.exists(generated_image_path):
                logger.info(f"📸 Processing generated image: {generated_image_path}")
                try:
                    with open(generated_image_path, "rb") as f:
                        image_bytes = f.read()
                    
                    if len(image_bytes) == 0:
                        raise ValueError("Generated image file is empty")
                    
                    # Determine filename and mime type
                    filename = "ai_generated_image.png"
                    mime_type = "image/png"
                    
                    # Check file extension for proper mime type
                    if generated_image_path.lower().endswith('.jpg') or generated_image_path.lower().endswith('.jpeg'):
                        mime_type = "image/jpeg"
                        filename = "ai_generated_image.jpg"
                    elif generated_image_path.lower().endswith('.webp'):
                        mime_type = "image/webp"
                        filename = "ai_generated_image.webp"
                    
                    # FIXED: Convert bytes to base64 string for FileWithBytes
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    artifacts.append(Part(root=FilePart(file=FileWithBytes(
                        bytes=image_b64,  # Now passing base64 string instead of raw bytes
                        mime_type=mime_type,
                        filename=filename
                    ))))
                    logger.info(f"✅ Image added to artifacts ({len(image_bytes)} bytes)")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process generated image: {e}")
                    # Don't fail the whole task, just log the error
                    artifacts.append(Part(root=TextPart(
                        text=f"Image was generated but couldn't be processed: {str(e)}"
                    )))
            
            # Ensure we have at least some response
            if not artifacts:
                if ai_response:
                    artifacts.append(Part(root=TextPart(text=ai_response)))
                else:
                    artifacts.append(Part(root=TextPart(
                        text="Image generation completed, but no output was captured. Please try again."
                    )))
            
            task_updater.add_artifact(artifacts)
            task_updater.complete()
            
            # Clean up temporary files
            #f generated_image_path and os.path.exists(generated_image_path):
               #try:
                #  os.remove(generated_image_path)
                 #  logger.info(f"🧹 Cleaned up temporary file: {generated_image_path}")
             #   except Exception as e:
           #         logger.warning(f"⚠️ Failed to clean up {generated_image_path}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Error processing image generation request: {e}")
            
            # Create user-friendly error message
            error_message = str(e)
            if "quota" in error_message.lower():
                error_message = "API quota exceeded. Please try again later or check your billing settings."
            elif "authentication" in error_message.lower() or "api key" in error_message.lower():
                error_message = "Authentication failed. Please check your API key configuration."
            elif "safety" in error_message.lower():
                error_message = "Content was filtered by safety guidelines. Please try a different prompt."
            elif "timeout" in error_message.lower():
                error_message = "Request timed out. Please try again with a simpler prompt."
            else:
                error_message = f"Image generation failed: {error_message}"
            
            # Add error as artifact so user can see it
            error_response = [Part(root=TextPart(text=f"❌ {error_message}"))]
            task_updater.add_artifact(error_response)
            task_updater.complete()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        """Executes the agent's logic based on the incoming A2A request."""
        logger.info(f"🚀 Starting execution for task {context.task_id}")
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            updater.submit()
        updater.start_work()
        
        await self._process_request(
            genai_types.UserContent(
                parts=self._convert_a2a_parts_to_genai(context.message.parts),
            ),
            context.context_id,
            updater,
            context.message.parts,  # Pass original A2A parts for text extraction
        )
        
        logger.info(f"🎉 Task {context.task_id} completed")

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Handle task cancellation."""
        logger.info(f"🛑 Cancelling task {context.task_id}")
        raise ServerError(error=UnsupportedOperationError())

    async def _upsert_session(self, session_id: str):
        """Create or retrieve existing session."""
        try:
            session = await self.runner.session_service.get_session(
                app_name=self.runner.app_name, 
                user_id='a2a_user', 
                session_id=session_id
            )
            if not session:
                session = await self.runner.session_service.create_session(
                    app_name=self.runner.app_name, 
                    user_id='a2a_user', 
                    session_id=session_id
                )
                logger.info(f"🆕 Created new session: {session_id}")
            else:
                logger.info(f"🔄 Using existing session: {session_id}")
            return session
        except Exception as e:
            logger.error(f"❌ Session management error: {e}")
            raise

    def _convert_a2a_parts_to_genai(self, parts: List[Part]) -> List[genai_types.Part]:
        """Converts a list of A2A Part objects to a list of Google GenAI Part objects."""
        return [self._convert_a2a_part_to_genai(part) for part in parts]

    def _convert_a2a_part_to_genai(self, part: Part) -> genai_types.Part:
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
                # FIXED: Handle both base64 string and raw bytes properly
                if isinstance(part.file.bytes, str):
                    # Already base64 encoded
                    data = base64.b64decode(part.file.bytes)
                else:
                    # Raw bytes - convert to base64 first
                    data = part.file.bytes
                
                return genai_types.Part(
                    inline_data=genai_types.Blob(
                        data=data, mime_type=part.file.mime_type
                    )
                )
            raise ValueError(f'Unsupported file type: {type(part.file)}')
        raise ValueError(f'Unsupported part type: {type(part)}')