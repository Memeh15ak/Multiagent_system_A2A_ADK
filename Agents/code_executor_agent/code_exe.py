#adk agent executor - FIXED VERSION
import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any
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
from Agents.code_executor_agent.code_executor import create_code_execution_agent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configuration to prevent infinite loops - REDUCED LIMITS
MAX_EVENTS_PER_SESSION = 10  # Reduced from 50 to 10
MAX_SESSION_TIME = 120  # Reduced from 300 to 120 seconds (2 minutes)
INTERIM_UPDATE_INTERVAL = 3  # Reduced from 5 to 3 seconds

class ADKCodeExecutionAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs an ADK-based Code Execution Agent with SINGLE FUNCTION CALL routing."""

    def __init__(self):
        # Initialize the ADK agent and runner.
        self._agent = create_code_execution_agent
        self.runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
        # Track active sessions to prevent loops
        self._active_sessions = {}

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
        """Processes the incoming request by running the ADK agent with STRICT SINGLE FUNCTION CALL."""
        
        # Initialize session tracking
        session_start_time = asyncio.get_event_loop().time()
        event_count = 0
        function_call_count = 0  # NEW: Track function calls
        has_final_response = False
        
        # FIX 1: Initialize last_interim_update to prevent UnboundLocalError
        last_interim_update = session_start_time
        
        # Track this session
        self._active_sessions[session_id] = {
            'start_time': session_start_time,
            'event_count': 0,
            'function_calls': 0,
            'task_updater': task_updater
        }
        
        try:
            session = await self._upsert_session(session_id)
            session_id = session.id
            
            logger.info(f"Starting ADK agent processing for session {session_id}")
            
            async for event in self._run_agent(session_id, new_message, task_updater):
                current_time = asyncio.get_event_loop().time()
                event_count += 1
                
                logger.debug(f'Received ADK event #{event_count}: {type(event).__name__}')
                
                # SAFETY CHECK 1: Maximum events per session (STRICT)
                if event_count > MAX_EVENTS_PER_SESSION:
                    logger.error(f"Session {session_id} exceeded maximum events ({MAX_EVENTS_PER_SESSION})")
                    # FIX 2: Use correct method name 'failed' instead of 'fail'
                    task_updater.failed(
                        error_message=f"Code execution terminated: Too many processing events ({event_count}). "
                                    f"Agent should use SINGLE function call only."
                    )
                    break
                
                # SAFETY CHECK 2: Maximum session time (STRICT)
                if (current_time - session_start_time) > MAX_SESSION_TIME:
                    logger.error(f"Session {session_id} exceeded maximum time ({MAX_SESSION_TIME}s)")
                    # FIX 2: Use correct method name 'failed' instead of 'fail'
                    task_updater.failed(
                        error_message=f"Code execution terminated: Time limit exceeded ({MAX_SESSION_TIME}s). "
                                    f"Agent should complete tasks faster with single function calls."
                    )
                    break
                
                # Update session tracking
                self._active_sessions[session_id]['event_count'] = event_count
                
                # Process the event
                if event.is_final_response():
                    logger.info(f"Received final response for session {session_id} after {event_count} events and {function_call_count} function calls")
                    # Convert ADK response to A2A artifact and complete the task
                    response_parts = convert_genai_parts_to_a2a(event.content.parts)
                    if response_parts:
                        task_updater.add_artifact(response_parts)
                    else:
                        # Ensure we have some response content
                        task_updater.add_artifact([TextPart(text="Code execution completed successfully")])
                    
                    task_updater.complete()
                    has_final_response = True
                    logger.info(f"Task completed successfully for session {session_id}")
                    break
                    
                elif event.get_function_calls():
                    # Handle function calls - COUNT THEM AND LIMIT
                    function_calls = event.get_function_calls()
                    function_call_count += len(function_calls)
                    
                    # NEW: STRICT LIMIT ON FUNCTION CALLS
                    if function_call_count > 1:
                        logger.error(f'Session {session_id} made {function_call_count} function calls - ONLY 1 ALLOWED')
                        # FIX 2: Use correct method name 'failed' instead of 'fail'
                        task_updater.failed(
                            error_message=f"Agent error: Made {function_call_count} function calls but should make ONLY 1. "
                                        f"Agent must use smart_execute_task for routing, not multiple functions."
                        )
                        break
                    
                    self._active_sessions[session_id]['function_calls'] = function_call_count
                    logger.info(f'Processing {len(function_calls)} function call(s) for session {session_id} (Total: {function_call_count})')
                    
                    # Log function call details with WARNING if not smart_execute_task
                    for i, func_call in enumerate(function_calls):
                        logger.info(f"Function call {i+1}: {func_call.name}")
                        if func_call.name != "smart_execute_task":
                            logger.warning(f"Agent called {func_call.name} directly instead of smart_execute_task - this causes multiple events!")
                    
                    # Provide interim update about function execution
                    if event.content.parts:
                        content_parts = convert_genai_parts_to_a2a(event.content.parts)
                        if content_parts:
                            task_updater.update_status(
                                TaskState.working,
                                message=task_updater.new_agent_message(content_parts),
                            )
                        
                elif event.content.parts:
                    # Regular content update (not final, no function calls)
                    logger.debug(f'Processing content update for session {session_id}')
                    
                    # Provide periodic interim updates to show progress
                    if (current_time - last_interim_update) >= INTERIM_UPDATE_INTERVAL:
                        content_parts = convert_genai_parts_to_a2a(event.content.parts)
                        if content_parts:
                            task_updater.update_status(
                                TaskState.working,
                                message=task_updater.new_agent_message(content_parts),
                            )
                            last_interim_update = current_time
                            logger.debug(f"Sent interim update for session {session_id}")
                
                else:
                    # Empty event - just log it
                    logger.debug(f'Received empty event for session {session_id}')
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.05)  # Reduced from 0.1 to 0.05
            
            # If we exit the loop without a final response, something went wrong
            if not has_final_response:
                logger.warning(f"Session {session_id} ended without final response after {event_count} events and {function_call_count} function calls")
                # FIX 2: Use correct method name 'failed' instead of 'fail'
                task_updater.failed(
                    error_message=f"Code execution incomplete: No final response received after {event_count} events and {function_call_count} function calls. "
                                f"Agent should complete tasks with single function call."
                )
        
        except Exception as e:
            logger.error(f"Error in _process_request for session {session_id}: {e}", exc_info=True)
            # FIX 2: Use correct method name 'failed' instead of 'fail'
            task_updater.failed(
                error_message=f"Code execution failed: {str(e)}"
            )
        
        finally:
            # Clean up session tracking
            if session_id in self._active_sessions:
                session_info = self._active_sessions[session_id]
                del self._active_sessions[session_id]
                logger.info(f"Session cleanup - Events: {session_info.get('event_count', 0)}, Function calls: {session_info.get('function_calls', 0)}")
            
            total_time = asyncio.get_event_loop().time() - session_start_time
            logger.info(f"Session {session_id} completed in {total_time:.2f}s with {event_count} events and {function_call_count} function calls")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        """Executes the agent's logic based on the incoming A2A request."""
        logger.info(f"Starting execution for task {context.task_id}")
        
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        # Initialize task if not already submitted
        if not context.current_task:
            updater.submit()
        
        # Start working on the task
        updater.start_work()
        
        try:
            # Convert A2A message to GenAI format
            genai_parts = convert_a2a_parts_to_genai(context.message.parts)
            genai_content = genai_types.UserContent(parts=genai_parts)
            
            # Process the request with STRICT loop prevention
            await self._process_request(
                genai_content,
                context.context_id,
                updater,
            )
            
        except Exception as e:
            logger.error(f"Error in execute for task {context.task_id}: {e}", exc_info=True)
            # FIX 2: Use correct method name 'failed' instead of 'fail'
            updater.failed(error_message=f"Execution failed: {str(e)}")

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Cancel the running task."""
        logger.info(f"Cancelling task {context.task_id}")
        
        # Clean up session if it's active
        if context.context_id in self._active_sessions:
            del self._active_sessions[context.context_id]
        
        # Note: ADK doesn't have built-in cancellation, so we just log and raise
        raise ServerError(error=UnsupportedOperationError())

    async def _upsert_session(self, session_id: str):
        """Retrieves or creates an ADK session."""
        try:
            session = await self.runner.session_service.get_session(
                app_name=self.runner.app_name, 
                user_id='self', 
                session_id=session_id
            )
            if not session:
                logger.info(f"Creating new session {session_id}")
                session = await self.runner.session_service.create_session(
                    app_name=self.runner.app_name, 
                    user_id='self', 
                    session_id=session_id
                )
            else:
                logger.info(f"Using existing session {session_id}")
            return session
        except Exception as e:
            logger.error(f"Error managing session {session_id}: {e}")
            raise

    def get_active_sessions_info(self):
        """Get information about currently active sessions (for debugging)."""
        current_time = asyncio.get_event_loop().time()
        info = {}
        for session_id, session_data in self._active_sessions.items():
            info[session_id] = {
                'runtime': current_time - session_data['start_time'],
                'event_count': session_data['event_count'],
                'function_calls': session_data.get('function_calls', 0)
            }
        return info

# Conversion functions with better error handling and logging
def convert_a2a_parts_to_genai(parts: list[Part]) -> list[genai_types.Part]:
    """Converts a list of A2A Part objects to a list of Google GenAI Part objects."""
    try:
        result = [convert_a2a_part_to_genai(part) for part in parts]
        logger.debug(f"Converted {len(parts)} A2A parts to {len(result)} GenAI parts")
        return result
    
    except Exception as e:
        logger.error(f"Error converting A2A parts to GenAI: {e}")
        # Return a simple text part as fallback
        return [genai_types.Part(text="Error processing message content")]

def convert_a2a_part_to_genai(part: Part) -> genai_types.Part:
    """Converts a single A2A Part object to a Google GenAI Part object."""
    try:
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
                return genai_types.Part(
                    inline_data=genai_types.Blob(
                        data=part_root.file.bytes, 
                        mime_type=part_root.file.mime_type
                    )
                )
            raise ValueError(f'Unsupported file type: {type(part_root.file)}')
        raise ValueError(f'Unsupported part type: {type(part_root)}')
    except Exception as e:
        logger.error(f"Error converting A2A part to GenAI: {e}")
        # Return a simple text part as fallback
        return genai_types.Part(text="Error processing content part")

def convert_genai_parts_to_a2a(parts: list[genai_types.Part]) -> list[Part]:
    """Converts a list of Google GenAI Part objects to a list of A2A Part objects."""
    try:
        result = []
        for part in parts:
            if part.text or part.file_data or part.inline_data:
                converted = convert_genai_part_to_a2a(part)
                if converted:
                    result.append(converted)
        logger.debug(f"Converted {len(parts)} GenAI parts to {len(result)} A2A parts")
        return result
    except Exception as e:
        logger.error(f"Error converting GenAI parts to A2A: {e}")
        # Return a simple text part as fallback
        return [Part(root=TextPart(text="Processing completed with errors"))]

def convert_genai_part_to_a2a(part: genai_types.Part) -> Part:
    """Converts a single Google GenAI Part object to an A2A Part object."""
    try:
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
                    bytes=part.inline_data.data,
                    mime_type=part.inline_data.mime_type,
                )
            ))
        logger.warning(f'Unsupported GenAI part type: {part}')
        return None
    except Exception as e:
        logger.error(f"Error converting GenAI part to A2A: {e}")
        return Part(root=TextPart(text="Error processing part"))