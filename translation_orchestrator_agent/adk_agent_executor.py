import asyncio
import logging
import os
import base64
from collections.abc import AsyncGenerator, AsyncIterable
from uuid import uuid4
from pathlib import Path

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
from a2a.utils.errors import ServerError
from translation_orchestrator_agent.adk_agent import create_smart_routing_agent
from translation_orchestrator_agent.dynamic_agent_discovery import SimpleAgentCardDiscovery

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Global multimedia context for sharing between executor and tools
_global_multimedia_context = {}

class ADKOrchestratorAgentExecutor(AgentExecutor):
    """Advanced ADK-based Orchestrator Agent Executor with complete agent support."""

    def __init__(self, temp_file_dir: str = "/tmp/a2a_files"):
        # Initialize ADK components
        self._agent = asyncio.run(create_smart_routing_agent())
        self.runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
        
        # Initialize agent discovery
        self.card_discovery = SimpleAgentCardDiscovery()
        asyncio.run(self._initialize_agent_cards())
        
        # Setup temp file directory
        self.temp_file_dir = Path(temp_file_dir)
        self.temp_file_dir.mkdir(parents=True, exist_ok=True)
        
        # Context tracking
        self._current_context_id = None
        self._current_multimedia_parts = []

    async def _initialize_agent_cards(self):
        """Initialize agent discovery system."""
        try:
            await self.card_discovery.extract_all_agent_cards()
            available = len([c for c in self.card_discovery.agent_cards.values() if c.available])
            total = len(self.card_discovery.agent_cards)
            logger.info(f"Agent cards initialized: {available}/{total} available")
        except Exception as e:
            logger.error(f"Failed to initialize agent cards: {e}")

    def _get_mime_type_info(self, mime_type: str) -> tuple[str, str]:
        """Get media type and file extension from MIME type."""
        if not mime_type or mime_type == 'None':
            return 'unknown', '.bin'
        
        mime_type = str(mime_type).lower()
        
        # Media type mapping
        if mime_type.startswith('image/'):
            media_type = 'image'
        elif mime_type.startswith('video/'):
            media_type = 'video'
        elif mime_type.startswith('audio/'):
            media_type = 'audio'
        elif mime_type.startswith('application/pdf'):
            media_type = 'document'
        elif mime_type.startswith('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'):
            media_type = 'excel'
        elif mime_type.startswith('application/vnd.ms-excel'):
            media_type = 'excel'
        elif mime_type.startswith('text/csv'):
            media_type = 'excel'
        elif mime_type.startswith('text/'):
            media_type = 'text'
        else:
            media_type = 'unknown'
        
        # Extension mapping
        extension_map = {
            'image/jpeg': '.jpg', 'image/png': '.png', 'image/gif': '.gif',
            'image/webp': '.webp', 'image/bmp': '.bmp', 'image/tiff': '.tiff',
            'video/mp4': '.mp4', 'video/avi': '.avi', 'video/mov': '.mov',
            'video/mkv': '.mkv', 'video/webm': '.webm', 'video/flv': '.flv',
            'audio/mp3': '.mp3', 'audio/mpeg': '.mp3', 'audio/wav': '.wav', 
            'audio/flac': '.flac', 'audio/ogg': '.ogg', 'audio/aac': '.aac',
            'application/pdf': '.pdf', 'text/plain': '.txt', 'text/markdown': '.md',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
            'application/vnd.ms-excel': '.xls', 'text/csv': '.csv',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        }
        extension = extension_map.get(mime_type, '.bin')
        
        return media_type, extension

    def _save_temp_file(self, file_data: bytes, mime_type: str, context_id: str) -> str:
        """Save file data to temporary path."""
        file_id = str(uuid4())
        _, extension = self._get_mime_type_info(mime_type)
        filename = f"{context_id}_{file_id}{extension}"
        file_path = self.temp_file_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        logger.info(f"Saved temp file: {file_path}")
        return str(file_path)

    async def _process_multimedia_content(self, parts: list[Part], context_id: str) -> tuple[str, list[dict]]:
        """Process multimedia content and extract user query."""
        user_query = ""
        file_paths = []
        
        for part in parts:
            if isinstance(part.root, TextPart):
                user_query = part.root.text
                
            elif isinstance(part.root, FilePart):
                file_obj = part.root.file
                mime_type = (
                    getattr(file_obj, 'mime_type', None) or 
                    getattr(file_obj, 'mimeType', None) or 
                    'application/octet-stream'
                )
                media_type, _ = self._get_mime_type_info(mime_type)
                
                if isinstance(file_obj, FileWithUri):
                    # Handle file URI
                    file_path = file_obj.uri
                    if file_path.startswith('file://'):
                        file_path = file_path[7:]
                    
                    # Ensure absolute path
                    file_path = os.path.abspath(os.path.normpath(file_path))
                    
                    file_info = {
                        'path': file_path,
                        'mime_type': mime_type,
                        'media_type': media_type,
                        'source': 'uri',
                        'exists': os.path.exists(file_path),
                        'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    }
                    file_paths.append(file_info)
                    logger.info(f"Added file URI: {file_path} (exists: {file_info['exists']})")
                    
                elif isinstance(file_obj, FileWithBytes):
                    # Handle file bytes
                    file_data = getattr(file_obj, 'data', None) or getattr(file_obj, 'bytes', None)
                    
                    if file_data:
                        # Save to temporary file
                        file_path = self._save_temp_file(file_data, mime_type, context_id)
                        
                        file_info = {
                            'path': file_path,
                            'mime_type': mime_type,
                            'media_type': media_type,
                            'source': 'bytes',
                            'exists': True,
                            'size': len(file_data)
                        }
                        file_paths.append(file_info)
                        logger.info(f"Added file bytes: {file_path} ({len(file_data)} bytes)")
        
        # Store context immediately after processing
        if file_paths:
            self._store_multimedia_context(context_id, file_paths)
            logger.info(f"Stored {len(file_paths)} files in context {context_id}")
        
        return user_query, file_paths

    def _create_enhanced_prompt(self, user_query: str, file_paths: list[dict]) -> str:
        """Create enhanced prompt with agent cards and file information."""
        try:
            # Get base prompt with agent cards
            enhanced_prompt = self.card_discovery.generate_adk_prompt_with_agent_cards(user_query)
            
            # Add multimedia file information if present
            if file_paths:
                file_info_lines = []
                
                for path_info in file_paths:
                    file_info_lines.append(
                        f"- {path_info['media_type']} file: {path_info['path']} "
                        f"(MIME: {path_info['mime_type']}, Size: {path_info['size']} bytes)"
                    )
                
                multimedia_context = f"""
[MULTIMEDIA FILES AVAILABLE]
The following files are available for processing:
{chr(10).join(file_info_lines)}

[CONTEXT INFORMATION]
- Files are saved as temporary paths and can be accessed directly
- Pass file paths directly to agent functions (do NOT convert to base64)
- Context ID: {self._current_context_id}

[USER REQUEST]
{user_query}
"""
                return multimedia_context
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Failed to create enhanced prompt: {e}")
            return user_query

    def _store_multimedia_context(self, context_id: str, file_paths: list[dict]):
        """Store multimedia context globally for tool access."""
        global _global_multimedia_context
        _global_multimedia_context[context_id] = {
            'multimedia_parts': file_paths,
            'timestamp': asyncio.get_event_loop().time()
        }
        logger.info(f"Stored multimedia context: {context_id} ({len(file_paths)} files)")

    async def _process_agent_events(self, session_id: str, message: genai_types.Content, 
                                  task_updater: TaskUpdater) -> None:
        """Process events from the ADK agent."""
        try:
            # Get the async generator properly
            event_stream = self.runner.run_async(
                session_id=session_id,
                user_id='self',
                new_message=message,
            )
            
            async for event in event_stream:
                logger.debug(f'Received ADK event: {event}')
                
                if event.is_final_response():
                    logger.info("Final response received from agent")
                    # Handle final response
                    if event.content and event.content.parts:
                        final_parts = convert_genai_parts_to_a2a(event.content.parts)
                        task_updater.add_artifact(parts=final_parts)
                    else:
                        # Fallback for empty response
                        default_part = Part(root=TextPart(text="No response content available"))
                        task_updater.add_artifact(parts=[default_part])
                    
                    task_updater.complete()
                    logger.info("Task completed successfully")
                    break
                    
                elif event.get_function_calls():
                    logger.info("Function calls detected in event")
                    # Store multimedia context when function calls are made
                    if self._current_multimedia_parts and self._current_context_id:
                        self._store_multimedia_context(self._current_context_id, self._current_multimedia_parts)
                    
                    function_calls = event.get_function_calls()
                    logger.info(f"Function calls generated: {function_calls}")
                    
                elif event.content and event.content.parts:
                    logger.info("Interim response with content parts")
                    # Handle interim responses
                    task_updater.update_status(
                        TaskState.working,
                        message=task_updater.new_agent_message(
                            convert_genai_parts_to_a2a(event.content.parts)
                        ),
                    )
                
                # Handle errors
                if hasattr(event, 'error_code') and event.error_code:
                    logger.error(f"Event error: {event.error_code}")
                    if event.error_code == 'MALFORMED_FUNCTION_CALL':
                        logger.error("Check agent tool definitions")
                        
        except Exception as e:
            logger.error(f"Error processing agent events: {e}")
            # Ensure task is marked as failed
            task_updater.fail(message=f"Agent execution failed: {str(e)}")

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Main execution method."""
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            updater.submit()
        updater.start_work()
        
        try:
            # Set current context
            self._current_context_id = context.context_id
            
            # Process multimedia content
            user_query, file_paths = await self._process_multimedia_content(
                context.message.parts, context.context_id
            )
            self._current_multimedia_parts = file_paths
            
            logger.info(f"Processing query: {user_query[:100]}...")
            if file_paths:
                logger.info(f"With {len(file_paths)} multimedia files")
            
            # Create enhanced prompt
            enhanced_prompt = self._create_enhanced_prompt(user_query, file_paths)
            
            # Create GenAI message
            user_content = genai_types.UserContent(
                parts=[genai_types.Part(text=enhanced_prompt)]
            )
            
            # Get or create session
            session = await self._get_or_create_session(context.context_id)
            
            # Process the request
            await self._process_agent_events(session.id, user_content, updater)
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            updater.fail(message=f"Execution failed: {str(e)}")

    async def _get_or_create_session(self, session_id: str):
        """Get existing session or create new one."""
        session = await self.runner.session_service.get_session(
            app_name=self.runner.app_name, 
            user_id='self', 
            session_id=session_id
        )
        
        if not session:
            session = await self.runner.session_service.create_session(
                app_name=self.runner.app_name, 
                user_id='self', 
                session_id=session_id
            )
        
        return session

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Cancel task execution."""
        raise ServerError(error=UnsupportedOperationError())

    def cleanup_temp_files(self, context_id: str = None):
        """Clean up temporary files."""
        if context_id:
            # Clean specific context files
            pattern = f"{context_id}_*"
            for file_path in self.temp_file_dir.glob(pattern):
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up: {file_path}")
                except Exception as e:
                    logger.error(f"Cleanup failed {file_path}: {e}")
        else:
            # Clean old files (>1 hour)
            import time
            current_time = time.time()
            for file_path in self.temp_file_dir.iterdir():
                try:
                    if file_path.is_file() and (current_time - file_path.stat().st_mtime) > 3600:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")
                except Exception as e:
                    logger.error(f"Old file cleanup failed {file_path}: {e}")


# Global helper functions for tools
def get_multimedia_context_for_request(context_id: str = None) -> list:
    """Get multimedia file paths for tools."""
    global _global_multimedia_context
    
    if context_id and context_id in _global_multimedia_context:
        context_data = _global_multimedia_context[context_id]
        logger.info(f"Retrieved context {context_id}: {len(context_data['multimedia_parts'])} files")
        return context_data['multimedia_parts']
    
    # Fallback to latest context
    if _global_multimedia_context:
        latest_context = max(_global_multimedia_context.items(), key=lambda x: x[1]['timestamp'])
        logger.info(f"Using latest context: {len(latest_context[1]['multimedia_parts'])} files")
        return latest_context[1]['multimedia_parts']
    
    return []

def get_file_paths_from_context(context_id: str = None) -> list[str]:
    """Get just file paths from context."""
    multimedia_parts = get_multimedia_context_for_request(context_id)
    return [part['path'] for part in multimedia_parts if 'path' in part]

def get_file_info_from_context(context_id: str = None, media_type: str = None) -> list[dict]:
    """Get file info, optionally filtered by media type."""
    multimedia_parts = get_multimedia_context_for_request(context_id)
    
    if media_type:
        return [part for part in multimedia_parts if part.get('media_type') == media_type]
    return multimedia_parts

def clear_multimedia_context(context_id: str = None):
    """Clean up multimedia context."""
    global _global_multimedia_context
    
    if context_id and context_id in _global_multimedia_context:
        del _global_multimedia_context[context_id]
        logger.info(f"Cleared context: {context_id}")
    elif context_id is None:
        # Clear contexts older than 5 minutes
        import time
        current_time = time.time()
        expired = [
            cid for cid, data in _global_multimedia_context.items()
            if current_time - data['timestamp'] > 300
        ]
        for cid in expired:
            del _global_multimedia_context[cid]
        logger.info(f"Cleared {len(expired)} expired contexts")


# Conversion utilities
def convert_genai_parts_to_a2a(parts: list[genai_types.Part]) -> list[Part]:
    """Convert GenAI parts to A2A parts."""
    return [
        convert_genai_part_to_a2a(part)
        for part in parts
        if (part.text or part.file_data or part.inline_data)
    ]

def convert_genai_part_to_a2a(part: genai_types.Part) -> Part:
    """Convert single GenAI part to A2A part."""
    if part.text:
        return Part(root=TextPart(text=part.text))
    if part.file_data:
        return Part(root=FilePart(
            file=FileWithUri(
                uri=part.file_data.file_uri,
                mimeType=part.file_data.mime_type,
            )
        ))
    if part.inline_data:
        return Part(root=FilePart(
            file=FileWithBytes(
                data=part.inline_data.data,
                mimeType=part.inline_data.mime_type,
            )
        ))
    raise ValueError(f'Unsupported part type: {part}')

def convert_a2a_parts_to_genai(parts: list[Part]) -> list[genai_types.Part]:
    """Convert A2A parts to GenAI parts."""
    return [convert_a2a_part_to_genai(part) for part in parts]

def convert_a2a_part_to_genai(part: Part) -> genai_types.Part:
    """Convert single A2A part to GenAI part."""
    part = part.root
    if isinstance(part, TextPart):
        return genai_types.Part(text=part.text)
    if isinstance(part, FilePart):
        if isinstance(part.file, FileWithUri):
            mime_type = getattr(part.file, 'mime_type', None) or getattr(part.file, 'mimeType', 'application/octet-stream')
            return genai_types.Part(
                file_data=genai_types.FileData(
                    file_uri=part.file.uri, 
                    mime_type=mime_type
                )
            )
        if isinstance(part.file, FileWithBytes):
            mime_type = getattr(part.file, 'mime_type', None) or getattr(part.file, 'mimeType', 'application/octet-stream')
            data = getattr(part.file, 'bytes', None) or getattr(part.file, 'data', None)
            if data is None:
                raise ValueError('FileWithBytes object has no data attribute')
            
            return genai_types.Part(
                inline_data=genai_types.Blob(
                    data=data, 
                    mime_type=mime_type
                )
            )
        raise ValueError(f'Unsupported file type: {type(part.file)}')
    raise ValueError(f'Unsupported part type: {type(part)}')