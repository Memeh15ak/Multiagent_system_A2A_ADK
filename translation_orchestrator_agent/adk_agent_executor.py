import asyncio
import logging
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
import base64
import os
from pathlib import Path
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
from translation_orchestrator_agent.adk_agent import create_smart_routing_agent
from translation_orchestrator_agent.dynamic_agent_discovery import SimpleAgentCardDiscovery


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Global multimedia context for sharing between executor and tools
_global_multimedia_context = {}

class ADKOrchestratorAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs an ADK-based Research Orchestrator Agent with enhanced multimedia support."""

    def __init__(self, temp_file_dir: str = "/tmp/a2a_files"):
        # Initialize the ADK agent and runner for the orchestrator.
        self._agent = asyncio.run(create_smart_routing_agent())
        self.runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
        # Store the current multimedia context for tool calls
        self._current_multimedia_parts = []
        self._current_context_id = None
        # Initialize agent discovery
        self.card_discovery = SimpleAgentCardDiscovery()
        # Initialize temp file directory
        self.temp_file_dir = Path(temp_file_dir)
        self.temp_file_dir.mkdir(parents=True, exist_ok=True)
        # Extract agent cards at initialization
        asyncio.run(self._initialize_agent_cards())

    async def _initialize_agent_cards(self):
        """Initialize agent cards discovery."""
        try:
            await self.card_discovery.extract_all_agent_cards()
            available_count = len([c for c in self.card_discovery.agent_cards.values() if c.available])
            total_count = len(self.card_discovery.agent_cards)
            logger.info(f"ADK Executor: Agent cards initialized. Available: {available_count}/{total_count}")
        except Exception as e:
            logger.error(f"Failed to initialize agent cards: {e}")

    def _save_file_to_temp_path(self, file_data: bytes, mime_type: str, context_id: str) -> str:
        """Save file data to a temporary path and return the path."""
        # Generate a unique filename
        file_id = str(uuid4())
        
        # Get appropriate file extension based on mime type
        extension = self._get_file_extension(mime_type)
        filename = f"{context_id}_{file_id}{extension}"
        
        # Save to temp directory
        file_path = self.temp_file_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        logger.info(f"Saved file to temporary path: {file_path}")
        return str(file_path)

    def _get_file_extension(self, mime_type: str) -> str:
        """Get file extension based on MIME type."""
        extension_map = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp',
            'image/bmp': '.bmp',
            'image/tiff': '.tiff',
            'video/mp4': '.mp4',
            'video/avi': '.avi',
            'video/mov': '.mov',
            'video/mkv': '.mkv',
            'audio/mp3': '.mp3',
            'audio/wav': '.wav',
            'audio/flac': '.flac',
            'audio/ogg': '.ogg',
            'application/pdf': '.pdf',
            'text/plain': '.txt',
        }
        return extension_map.get(mime_type, '.bin')

    def _run_agent(
        self,
        session_id: str,
        new_message: genai_types.Content,
        task_updater: TaskUpdater,
    ) -> AsyncGenerator[Event, None]:
        """Runs the ADK orchestrator agent with the given message."""
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
        """Processes the incoming request by running the ADK orchestrator agent."""
        session = await self._upsert_session(session_id)
        session_id = session.id
        
        async for event in self._run_agent(session_id, new_message, task_updater):
            logger.debug('Orchestrator Received ADK event: %s', event)
            
            if event.is_final_response():
                # FIX: Check if event.content exists before accessing parts
                if event.content and event.content.parts:
                    final_parts = convert_genai_parts_to_a2a(event.content.parts)
                    logger.debug('Orchestrator LLM final content parts: %s', final_parts)
                    task_updater.add_artifact(parts=final_parts)
                else:
                    logger.warning("Final response event has no content or parts")
                    # Create a default empty response
                    from a2a.types import TextPart, Part
                    default_part = Part(root=TextPart(text="No response content available"))
                    task_updater.add_artifact(parts=[default_part])
                
                task_updater.complete()
                logger.info("Orchestrator task completed with final parts added as artifact.")
                break
                
            elif event.get_function_calls():
                # CRITICAL FIX: Store multimedia context globally when function calls are made
                function_calls = event.get_function_calls()
                logger.info(f"Orchestrator LLM generated function call: {function_calls}")
                
                # Store multimedia context globally for the tools to access
                if self._current_multimedia_parts and self._current_context_id:
                    global _global_multimedia_context
                    _global_multimedia_context[self._current_context_id] = {
                        'multimedia_parts': self._current_multimedia_parts,
                        'timestamp': asyncio.get_event_loop().time()
                    }
                    logger.info(f"Stored multimedia context globally for context_id: {self._current_context_id}")
                    logger.info(f"Tool call will have access to {len(self._current_multimedia_parts)} multimedia parts")
                    
            elif event.content and event.content.parts:
                # Interim response from the orchestrator's LLM
                logger.debug('Orchestrator LLM interim response parts: %s', event.content.parts)
                task_updater.update_status(
                    TaskState.working,
                    message=task_updater.new_agent_message(
                        convert_genai_parts_to_a2a(event.content.parts)
                    ),
                )
            else:
                logger.debug('Orchestrator skipping event: %s', event)
                # FIX: Also handle the case where event has an error
                if hasattr(event, 'error_code') and event.error_code:
                    logger.error(f"Event error: {event.error_code} - {getattr(event, 'error_message', 'No error message')}")
                    # You might want to handle specific error cases here
                    if event.error_code == 'MALFORMED_FUNCTION_CALL':
                        logger.error("Malformed function call detected - check agent tool definitions")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        """Executes the orchestrator agent's logic based on the incoming A2A request."""
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            updater.submit()
        updater.start_work()
        
        # Store context ID for global multimedia context sharing
        self._current_context_id = context.context_id
        
        # Enhanced multimedia content handling - now with direct path passing
        initial_user_message_parts, file_paths = await self._process_multimedia_content(context.message.parts, context.context_id)
        
        # Store multimedia file paths for tool access
        self._current_multimedia_parts = file_paths
        
        # Extract the user's query text
        user_query = ""
        for part in context.message.parts:
            if hasattr(part.root, 'text') and part.root.text:
                user_query = part.root.text
                break
        
        logger.info(f"Processing user query: {user_query[:100]}...")
        
        # CRITICAL FIX: Generate enhanced prompt with agent discovery
        try:
            enhanced_prompt = self.card_discovery.generate_adk_prompt_with_agent_cards(user_query)
            logger.info(f"Generated enhanced prompt with agent cards ({len(enhanced_prompt)} chars)")
            
            # Create enhanced prompt with file path information
            if file_paths:
                file_info = []
                for path_info in file_paths:
                    file_info.append(f"- {path_info['media_type']} file: {path_info['path']} (MIME: {path_info['mime_type']})")
                
                multimedia_context = f"""
[MULTIMEDIA FILES AVAILABLE]
The following files are available for processing:
{chr(10).join(file_info)}

[IMPORTANT ROUTING INSTRUCTIONS]
- Files are saved as temporary paths and can be accessed directly by agents
- Pass the file paths directly to the appropriate agent functions
- Do NOT convert files to base64 - use the paths directly
- Route to appropriate agents based on file types:
  * Images: Use image_modification_function
  * Audio: Use audio_conversational_function  
  * Video: Use video_function

[USER REQUEST]
{user_query}
"""
                enhanced_prompt = multimedia_context
            
            # Create text part with enhanced prompt
            enhanced_parts = [genai_types.Part(text=enhanced_prompt)]
            
            # Create enhanced message (text only, no binary data)
            user_content = genai_types.UserContent(parts=enhanced_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced prompt: {e}")
            # Fallback to original text-only message
            user_content = genai_types.UserContent(parts=[genai_types.Part(text=user_query)])
        
        await self._process_request(
            user_content,
            context.context_id,
            updater,
        )

    async def _process_multimedia_content(self, parts: list[Part], context_id: str) -> tuple[list[genai_types.Part], list[dict]]:
        """Process multimedia content by saving files to temporary paths."""
        text_parts = []
        file_paths = []
        
        for part in parts:
            if isinstance(part.root, TextPart):
                text_parts.append(genai_types.Part(text=part.root.text))
                
            elif isinstance(part.root, FilePart):
                file_obj = part.root.file
                
                # Get MIME type
                mime_type = (
                    getattr(file_obj, 'mime_type', None) or 
                    getattr(file_obj, 'mimeType', None) or 
                    getattr(file_obj, 'content_type', None) or
                    'application/octet-stream'
                )
                
                if isinstance(file_obj, FileWithUri):
                    file_path = file_obj.uri
                    
                    # Remove file:// prefix if present
                    if file_path.startswith('file://'):
                        file_path = file_path[7:]  # Remove 'file://' prefix
                    
                    # Normalize the path
                    normalized_path = os.path.normpath(file_path)
                    
                    # Verify the file exists
                    if os.path.exists(normalized_path):
                        file_info = {
                            'path': normalized_path,
                            'mime_type': mime_type,
                            'media_type': self._get_media_type(mime_type),
                            'source': 'uri'
                        }
                        file_paths.append(file_info)
                        logger.info(f"Added file path: {normalized_path}")
                    else:
                        logger.error(f"File does not exist: {normalized_path}")
                        file_info = {
                            'path': normalized_path,
                            'mime_type': mime_type,
                            'media_type': self._get_media_type(mime_type),
                            'source': 'uri',
                            'exists': False
                        }
                        file_paths.append(file_info)
                        
                elif isinstance(file_obj, FileWithBytes):
                    # For bytes files, save to temporary path
                    data = (
                        getattr(file_obj, 'bytes', None) or 
                        getattr(file_obj, 'data', None)
                    )
                    
                    if data is None:
                        logger.error(f"FileWithBytes object has no data attribute")
                        continue
                    
                    # Handle base64 encoded data
                    if isinstance(data, str):
                        try:
                            decoded_data = base64.b64decode(data)
                            logger.info(f"Decoded base64 data: {len(decoded_data)} bytes")
                        except Exception as e:
                            logger.error(f"Failed to decode base64: {e}")
                            continue
                    else:
                        decoded_data = data
                        logger.info(f"Using raw bytes data: {len(decoded_data)} bytes")
                    
                    # Save to temporary file
                    temp_path = self._save_file_to_temp_path(decoded_data, mime_type, context_id)
                    
                    file_info = {
                        'path': temp_path,
                        'mime_type': mime_type,
                        'media_type': self._get_media_type(mime_type),
                        'source': 'bytes',
                        'size': len(decoded_data)
                    }
                    file_paths.append(file_info)
                    logger.info(f"Saved bytes file to: {temp_path}")
        
        return text_parts, file_paths

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Cancel the current task execution."""
        raise ServerError(error=UnsupportedOperationError())

    async def _upsert_session(self, session_id: str):
        """Retrieves or creates an ADK session for the orchestrator."""
        return await self.runner.session_service.get_session(
            app_name=self.runner.app_name, user_id='self', session_id=session_id
        ) or await self.runner.session_service.create_session(
            app_name=self.runner.app_name, user_id='self', session_id=session_id
        )

    def _get_media_type(self, mime_type: str) -> str:
        """Determine the media type from MIME type."""
        # Handle None case explicitly
        if mime_type is None or mime_type == 'None':
            return 'unknown'
            
        # Ensure mime_type is a string
        mime_type = str(mime_type).lower()
        
        if mime_type.startswith('image/'):
            return 'image'
        elif mime_type.startswith('video/'):
            return 'video'
        elif mime_type.startswith('audio/'):
            return 'audio'
        elif mime_type.startswith('application/pdf'):
            return 'document'
        elif mime_type.startswith('text/'):
            return 'text'
        else:
            return 'unknown'

    def cleanup_temp_files(self, context_id: str = None):
        """Clean up temporary files after processing."""
        if context_id:
            # Clean up files for specific context
            pattern = f"{context_id}_*"
            for file_path in self.temp_file_dir.glob(pattern):
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up temp file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to clean up temp file {file_path}: {e}")
        else:
            # Clean up all old temp files (older than 1 hour)
            import time
            current_time = time.time()
            for file_path in self.temp_file_dir.iterdir():
                try:
                    if file_path.is_file() and (current_time - file_path.stat().st_mtime) > 3600:
                        file_path.unlink()
                        logger.info(f"Cleaned up old temp file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to clean up old temp file {file_path}: {e}")

# Global helper functions for tools to access multimedia context
def get_multimedia_context_for_request(context_id: str = None) -> list:
    """Helper function for tools to retrieve multimedia file paths."""
    global _global_multimedia_context
    
    if context_id and context_id in _global_multimedia_context:
        context_data = _global_multimedia_context[context_id]
        logger.info(f"Retrieved multimedia context for {context_id}: {len(context_data['multimedia_parts'])} files")
        return context_data['multimedia_parts']
    
    # Fallback: return the most recent context (for backwards compatibility)
    if _global_multimedia_context:
        latest_context = max(_global_multimedia_context.items(), key=lambda x: x[1]['timestamp'])
        logger.info(f"Using latest multimedia context: {len(latest_context[1]['multimedia_parts'])} files")
        return latest_context[1]['multimedia_parts']
    
    logger.info("No multimedia context available")
    return []

def get_file_paths_from_context(context_id: str = None) -> list[str]:
    """Get just the file paths from multimedia context."""
    multimedia_parts = get_multimedia_context_for_request(context_id)
    return [part['path'] for part in multimedia_parts if 'path' in part]

def get_file_info_from_context(context_id: str = None, media_type: str = None) -> list[dict]:
    """Get file info from multimedia context, optionally filtered by media type."""
    multimedia_parts = get_multimedia_context_for_request(context_id)
    
    if media_type:
        return [part for part in multimedia_parts if part.get('media_type') == media_type]
    
    return multimedia_parts

def clear_multimedia_context(context_id: str = None):
    """Clean up multimedia context after request processing."""
    global _global_multimedia_context
    
    if context_id and context_id in _global_multimedia_context:
        del _global_multimedia_context[context_id]
        logger.info(f"Cleared multimedia context for {context_id}")
    elif context_id is None:
        # Clear all contexts older than 5 minutes
        import time
        current_time = time.time()
        expired_contexts = [
            cid for cid, data in _global_multimedia_context.items()
            if current_time - data['timestamp'] > 300  # 5 minutes
        ]
        for cid in expired_contexts:
            del _global_multimedia_context[cid]
        logger.info(f"Cleared {len(expired_contexts)} expired multimedia contexts")

# Conversion functions for backwards compatibility
def convert_a2a_parts_to_genai(parts: list[Part]) -> list[genai_types.Part]:
    """Standard conversion function for backwards compatibility."""
    return [convert_a2a_part_to_genai(part) for part in parts]


def convert_a2a_part_to_genai(part: Part) -> genai_types.Part:
    """Standard conversion function for backwards compatibility."""
    part = part.root
    if isinstance(part, TextPart):
        return genai_types.Part(text=part.text)
    if isinstance(part, FilePart):
        if isinstance(part.file, FileWithUri):
            # Try both attribute names for mime_type
            mime_type = getattr(part.file, 'mime_type', None) or getattr(part.file, 'mimeType', 'application/octet-stream')
            return genai_types.Part(
                file_data=genai_types.FileData(
                    file_uri=part.file.uri, 
                    mime_type=mime_type
                )
            )
        if isinstance(part.file, FileWithBytes):
            # Try both attribute names for mime_type and data
            mime_type = getattr(part.file, 'mime_type', None) or getattr(part.file, 'mimeType', 'application/octet-stream')
            data = getattr(part.file, 'bytes', None) or getattr(part.file, 'data', None)
            if data is None:
                raise ValueError(f'FileWithBytes object has no data attribute')
            
            return genai_types.Part(
                inline_data=genai_types.Blob(
                    data=data, 
                    mime_type=mime_type
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
        return Part(root=TextPart(text=part.text))
    if part.file_data:
        return Part(root=FilePart(
            file=FileWithUri(
                uri=part.file_data.file_uri,
                mimeType=part.file_data.mime_type,  # Use mimeType for consistency
            )
        ))
    if part.inline_data:
        return Part(
            root=FilePart(
                file=FileWithBytes(
                    data=part.inline_data.data,
                    mimeType=part.inline_data.mime_type,  # Use mimeType for consistency
                )
            )
        )
    raise ValueError(f'Unsupported part type: {part}')