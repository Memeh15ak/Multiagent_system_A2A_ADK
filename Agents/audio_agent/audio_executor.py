import logging
import tempfile
import os
import base64
from collections.abc import AsyncGenerator, AsyncIterable
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

# Import the audio agent creator from your audio_agent.py file
from Agents.audio_agent.audio_adk import create_audio_agent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ADKAudioAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs an ADK-based Audio Processing Agent."""

    def __init__(self):
        # Don't initialize the agent here - do it lazily when needed
        self._agent = None
        self.runner = None

    async def _get_agent_and_runner(self):
        """Lazily initialize the agent and runner when first needed."""
        if self._agent is None:
            # Create the audio agent
            self._agent = await create_audio_agent()
            self.runner = Runner(
                app_name=self._agent.name,
                agent=self._agent,
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

    async def _process_audio_file(self, file_part: FilePart) -> str:
        """Process audio file and save to temporary location."""
        try:
            if isinstance(file_part.file, FileWithBytes):
                # Handle bytes properly - check if it's base64 encoded string or raw bytes
                file_bytes = file_part.file.bytes
                
                if isinstance(file_bytes, str):
                    # If it's a string, decode from base64
                    try:
                        file_bytes = base64.b64decode(file_bytes)
                        logger.info(f"Decoded base64 string to {len(file_bytes)} bytes")
                    except Exception as e:
                        logger.error(f"Failed to decode base64 string: {e}")
                        raise ServerError(f"Invalid base64 audio data: {e}")
                elif isinstance(file_bytes, bytes):
                    # Already bytes, use as is
                    logger.info(f"Using raw bytes: {len(file_bytes)} bytes")
                else:
                    raise ServerError(f"Unsupported file bytes type: {type(file_bytes)}")
                
                # Get proper file extension
                file_ext = self._get_file_extension(file_part.file.mime_type)
                temp_file_path = os.path.join(tempfile.gettempdir(), f"audio_{uuid4()}{file_ext}")
                
                # Save bytes to temporary file
                with open(temp_file_path, 'wb') as f:
                    f.write(file_bytes)
                
                logger.info(f"Saved audio file to: {temp_file_path} ({len(file_bytes)} bytes)")
                return temp_file_path
                
            elif isinstance(file_part.file, FileWithUri):
                # For URI-based files, return the URI
                logger.info(f"Using audio file URI: {file_part.file.uri}")
                return file_part.file.uri
            else:
                raise ServerError(f"Unsupported file type: {type(file_part.file)}")
                
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            raise ServerError(f"Failed to process audio file: {str(e)}")
    
    def _get_file_extension(self, mime_type: str) -> str:
        """Get file extension from mime type."""
        if not mime_type:
            return '.mp3'  # Default fallback
            
        mime_to_ext = {
            'audio/mpeg': '.mp3',
            'audio/mp3': '.mp3',
            'audio/wav': '.wav',
            'audio/x-wav': '.wav',
            'audio/wave': '.wav',
            'audio/mp4': '.m4a',
            'audio/x-m4a': '.m4a',
            'audio/flac': '.flac',
            'audio/aac': '.aac',
            'audio/ogg': '.ogg',
            'audio/webm': '.webm',
            'audio/x-ms-wma': '.wma',
            'audio/opus': '.opus'
        }
        return mime_to_ext.get(mime_type.lower(), '.mp3')

    async def _create_audio_response(self, response_text: str) -> Part:
        """Create an audio response from text using the agent's audio response function."""
        try:
            # Get the agent to use its audio_response_function
            agent, runner = await self._get_agent_and_runner()
            
            # Import the audio response function directly
            from Agents.audio_agent.audio_adk import audio_response_function
            
            # Generate audio using the agent's function
            audio_result = audio_response_function(response_text, "natural")
            
            if audio_result.get('status') == 'success':
                # Get the audio bytes from base64
                audio_b64 = audio_result.get('audio_bytes_b64')
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    mime_type = audio_result.get('mime_type', 'audio/wav')
                    
                    # Create FilePart with proper bytes handling
                    return Part(
                        root=FilePart(
                            file=FileWithBytes(
                                bytes=audio_bytes,  # Use raw bytes, not base64 string
                                mime_type=mime_type
                            )
                        )
                    )
                else:
                    # Fallback: read from file path if available
                    audio_file_path = audio_result.get('audio_file_path')
                    if audio_file_path and os.path.exists(audio_file_path):
                        with open(audio_file_path, 'rb') as f:
                            audio_bytes = f.read()
                        
                        return Part(
                            root=FilePart(
                                file=FileWithBytes(
                                    bytes=audio_bytes,
                                    mime_type='audio/wav'
                                )
                            )
                        )
            
            # If audio generation failed, return text response
            logger.warning("Audio generation failed, returning text response")
            return Part(root=TextPart(text=response_text))
            
        except Exception as e:
            logger.error(f"Error creating audio response: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Fallback to text response if audio generation fails
            return Part(root=TextPart(text=f"Audio generation failed: {str(e)}. Text response: {response_text}"))

    async def _process_request(
        self,
        new_message: genai_types.Content,
        session_id: str,
        task_updater: TaskUpdater,
    ) -> AsyncIterable[TaskStatus | Artifact]:
        """Processes the incoming request by running the ADK audio agent."""
        # Ensure agent and runner are initialized
        await self._get_agent_and_runner()
        
        session = await self._upsert_session(session_id)
        session_id = session.id
        
        # Process any audio files in the message
        audio_files = []
        text_parts = []
        
        for part in new_message.parts:
            if part.file_data or part.inline_data:
                # This is a file part - check if it's audio
                mime_type = (
                    getattr(part.file_data, 'mime_type', None) or 
                    getattr(part.inline_data, 'mime_type', None)
                )
                
                if mime_type and mime_type.startswith('audio/'):
                    # Convert to FilePart and process
                    try:
                        file_part = convert_genai_part_to_a2a_file(part)
                        audio_file_path = await self._process_audio_file(file_part)
                        audio_files.append(audio_file_path)
                        logger.info(f"Successfully processed audio file: {audio_file_path}")
                    except Exception as e:
                        logger.error(f"Failed to process audio file: {e}")
                        text_parts.append(f"Error processing audio file: {e}")
                        
            elif part.text:
                text_parts.append(part.text)
        
        # Create enhanced message with audio file information
        if audio_files:
            audio_info = f"Audio files to process: {', '.join(audio_files)}"
            text_query = ' '.join(text_parts) if text_parts else "Please process the uploaded audio file."
            enhanced_message = f"{audio_info}\nUser query: {text_query}"
            logger.info(f"Processing {len(audio_files)} audio files with query: {text_query}")
        else:
            enhanced_message = ' '.join(text_parts) if text_parts else "No audio files provided."
            logger.warning("No audio files found in the request")
        
        # Create new message for the agent
        agent_message = genai_types.UserContent(
            parts=[genai_types.Part(text=enhanced_message)]
        )
        
        # Track if we've received a final response
        final_response_received = False
        
        try:
            async for event in self._run_agent(session_id, agent_message, task_updater):
                logger.debug('Received ADK event: %s', event)
                
                if event.is_final_response() and not final_response_received:
                    final_response_received = True
                    
                    # Extract text response from the event
                    response_text = self._extract_text_from_event(event)
                    logger.info(f"Final response received: {response_text[:100]}...")
                    
                    # Create both text and audio responses
                    response_parts = [
                        Part(root=TextPart(text=response_text))
                    ]
                    
                    # Only create audio response if we have substantial text and audio files were processed
                    if len(response_text.strip()) > 10 and audio_files:
                        try:
                            audio_response = await self._create_audio_response(response_text)
                            response_parts.append(audio_response)
                            logger.info("Audio response created successfully")
                        except Exception as e:
                            logger.error(f"Failed to create audio response: {e}")
                            # Continue with just text response
                    
                    task_updater.add_artifact(response_parts)
                    task_updater.complete()
                    break
                    
                elif not event.get_function_calls():
                    # Interim update - only if we haven't sent final response
                    if not final_response_received:
                        response_text = self._extract_text_from_event(event)
                        if response_text.strip():  # Only update if there's actual content
                            task_updater.update_status(
                                TaskState.working,
                                message=task_updater.new_agent_message([
                                    Part(root=TextPart(text=f"Processing: {response_text}"))
                                ]),
                            )
                else:
                    # Handle function calls - only if we haven't sent final response
                    if not final_response_received:
                        logger.debug('Processing event with function call: %s', event.get_function_calls())
                        task_updater.update_status(
                            TaskState.working,
                            message=task_updater.new_agent_message([
                                Part(root=TextPart(text="Processing audio with specialized tools..."))
                            ]),
                        )
                        
        except Exception as e:
            logger.error(f"Error during agent processing: {e}")
            if not final_response_received:
                task_updater.add_artifact([
                    Part(root=TextPart(text=f"Error during audio processing: {str(e)}"))
                ])
                task_updater.complete()

    def _extract_text_from_event(self, event: Event) -> str:
        """Extract text content from ADK event."""
        text_parts = []
        
        if hasattr(event, 'content') and event.content:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
        
        result = ' '.join(text_parts) if text_parts else ""
        
        # Clean up the text - remove debug markers and formatting
        result = result.strip()
        if not result:
            result = "Audio processing completed."
        
        return result

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
        
        try:
            # Convert A2A message to GenAI format
            genai_message = genai_types.UserContent(
                parts=convert_a2a_parts_to_genai(context.message.parts),
            )
            
            logger.info(f"Executing audio agent with {len(context.message.parts)} parts")
            
            await self._process_request(
                genai_message,
                context.context_id,
                updater,
            )
        except Exception as e:
            logger.error(f"Error in execute: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Send error response
            updater.add_artifact([
                Part(root=TextPart(text=f"Error processing request: {str(e)}"))
            ])
            updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())

    async def _upsert_session(self, session_id: str):
        """Retrieves or creates an ADK session."""
        if not self.runner:
            await self._get_agent_and_runner()
            
        return await self.runner.session_service.get_session(
            app_name=self.runner.app_name, user_id='self', session_id=session_id
        ) or await self.runner.session_service.create_session(
            app_name=self.runner.app_name, user_id='self', session_id=session_id
        )


def convert_a2a_parts_to_genai(parts: list[Part]) -> list[genai_types.Part]:
    """Converts a list of A2A Part objects to a list of Google GenAI Part objects."""
    converted_parts = []
    for part in parts:
        try:
            converted = convert_a2a_part_to_genai(part)
            converted_parts.append(converted)
        except Exception as e:
            logger.error(f"Failed to convert part {part}: {e}")
            # Add error as text part instead of failing completely
            converted_parts.append(genai_types.Part(text=f"Error converting part: {e}"))
    return converted_parts


def convert_a2a_part_to_genai(part: Part) -> genai_types.Part:
    """Converts a single A2A Part object to a Google GenAI Part object."""
    part_root = part.root
    
    if isinstance(part_root, TextPart):
        return genai_types.Part(text=part_root.text)
        
    if isinstance(part_root, FilePart):
        if isinstance(part_root.file, FileWithUri):
            return genai_types.Part(
                file_data=genai_types.FileData(
                    file_uri=part_root.file.uri, 
                    mime_type=part_root.file.mime_type
                )
            )
        if isinstance(part_root.file, FileWithBytes):
            # Handle bytes properly
            file_bytes = part_root.file.bytes
            
            if isinstance(file_bytes, str):
                # If it's a string, assume it's base64 encoded
                try:
                    file_bytes = base64.b64decode(file_bytes)
                except Exception as e:
                    logger.error(f"Failed to decode base64 file data: {e}")
                    raise ValueError(f"Invalid base64 file data: {e}")
            elif not isinstance(file_bytes, bytes):
                raise ValueError(f"File bytes must be bytes or base64 string, got {type(file_bytes)}")
            
            return genai_types.Part(
                inline_data=genai_types.Blob(
                    data=file_bytes, 
                    mime_type=part_root.file.mime_type or 'application/octet-stream'
                )
            )
        raise ValueError(f'Unsupported file type: {type(part_root.file)}')
    
    raise ValueError(f'Unsupported part type: {type(part_root)}')


def convert_genai_parts_to_a2a(parts: list[genai_types.Part]) -> list[Part]:
    """Converts a list of Google GenAI Part objects to a list of A2A Part objects."""
    converted_parts = []
    for part in parts:
        try:
            if part.text or part.file_data or part.inline_data:
                converted = convert_genai_part_to_a2a(part)
                converted_parts.append(converted)
        except Exception as e:
            logger.error(f"Failed to convert GenAI part: {e}")
            # Convert to text part with error message
            converted_parts.append(Part(root=TextPart(text=f"Error converting part: {e}")))
    return converted_parts


def convert_genai_part_to_a2a(part: genai_types.Part) -> Part:
    """Converts a single Google GenAI Part object to an A2A Part object."""
    if part.text:
        return Part(root=TextPart(text=part.text))
        
    if part.file_data:
        return Part(root=FilePart(
            file=FileWithUri(
                uri=part.file_data.file_uri,
                mime_type=part.file_data.mime_type,
            )
        ))
        
    if part.inline_data:
        return Part(root=FilePart(
            file=FileWithBytes(
                bytes=part.inline_data.data,  # Keep as bytes
                mime_type=part.inline_data.mime_type,
            )
        ))
        
    raise ValueError(f'Unsupported part type: {part}')


def convert_genai_part_to_a2a_file(part: genai_types.Part) -> FilePart:
    """Converts a GenAI Part to a FilePart specifically."""
    if part.file_data:
        return FilePart(
            file=FileWithUri(
                uri=part.file_data.file_uri,
                mime_type=part.file_data.mime_type,
            )
        )
    if part.inline_data:
        return FilePart(
            file=FileWithBytes(
                bytes=part.inline_data.data,  # Keep as bytes
                mime_type=part.inline_data.mime_type,
            )
        )
    raise ValueError(f'Part is not a file: {part}')