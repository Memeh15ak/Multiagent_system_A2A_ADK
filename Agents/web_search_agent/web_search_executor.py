# mypy: ignore-errors
import asyncio
import logging

from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any
from uuid import uuid4

from google.adk import Runner
from google.adk.agents import Agent, RunConfig
from google.adk.artifacts import InMemoryArtifactService
from google.adk.events import Event
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from pydantic import ConfigDict

from a2a.client import A2AClient
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

from Agents.web_search_agent.web_search import create_web_search_agent


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ADKWebSearchAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs an ADK-based Web Search Agent with improved response handling."""

    def __init__(self):
        self._agent = None
        self.runner = None

    async def _get_agent_and_runner(self):
        """Lazily initialize the agent and runner when first needed."""
        if self._agent is None:
            self._agent = await create_web_search_agent()
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

    async def _process_request(
        self,
        new_message: genai_types.Content,
        session_id: str,
        task_updater: TaskUpdater,
    ) -> AsyncIterable[TaskStatus | Artifact]:
        """Processes the incoming request by running the ADK agent with simplified response handling."""
        await self._get_agent_and_runner()
        
        session = await self._upsert_session(session_id)
        session_id = session.id
        
        # Collect all response content
        all_response_content = []
        final_response_found = False
        
        async for event in self._run_agent(session_id, new_message, task_updater):
            logger.debug('Received ADK event: %s', event)
            
            # Extract text content from any event that has it
            event_text = self._extract_text_from_event(event)
            
            if event_text:
                logger.info(f"📝 Extracted text from event: {event_text[:200]}...")
                all_response_content.append(event_text)
                
                # Update task status with current content
                response_parts = [TextPart(text=event_text)]
                task_updater.update_status(
                    TaskState.working,
                    message=task_updater.new_agent_message(response_parts),
                )
            
            # Check if this looks like a final response
            if self._is_likely_final_response(event, event_text):
                logger.info("✅ DETECTED LIKELY FINAL RESPONSE")
                final_response_found = True
                break
        
        # Combine all response content
        combined_response = "\n\n".join(all_response_content) if all_response_content else "Search completed successfully."
        
        logger.info(f"✅ FINAL COMBINED RESPONSE: {combined_response[:300]}...")
        
        # Create final artifact
        final_parts = [TextPart(text=combined_response)]
        task_updater.add_artifact(final_parts)
        task_updater.complete()

    def _extract_text_from_event(self, event: Event) -> str:
        """Extract text content from an event using multiple strategies."""
        text_content = ""
        
        try:
            # Strategy 1: Direct content parts
            if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_content += part.text + " "
                        logger.debug(f"Found text in part.text: {part.text[:100]}...")
            
            # Strategy 2: Function responses
            if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                for part in event.content.parts:
                    if hasattr(part, 'function_response') and part.function_response:
                        response_data = part.function_response.response
                        if isinstance(response_data, dict):
                            for key in ['search_results', 'result', 'response', 'content', 'answer']:
                                if key in response_data and response_data[key]:
                                    text_content += str(response_data[key]) + " "
                                    logger.debug(f"Found text in function_response[{key}]")
                                    break
                        elif isinstance(response_data, str):
                            text_content += response_data + " "
                            logger.debug(f"Found direct string in function_response")
            
            # Strategy 3: Event string representation parsing (fallback)
            if not text_content.strip():
                event_str = str(event)
                # Look for meaningful text patterns in the string representation
                import re
                
                # Pattern for text within quotes
                text_matches = re.findall(r"text='([^']{50,})'", event_str)
                if text_matches:
                    text_content = max(text_matches, key=len)  # Get the longest match
                    logger.debug(f"Extracted text via regex: {text_content[:100]}...")
        
        except Exception as e:
            logger.warning(f"Error extracting text from event: {e}")
        
        return text_content.strip()

    def _is_likely_final_response(self, event: Event, text_content: str) -> bool:
        """Determine if this event represents a final response."""
        try:
            # Check 1: Event has is_final_response method and returns True
            if hasattr(event, 'is_final_response') and callable(event.is_final_response):
                if event.is_final_response():
                    return True
            
            # Check 2: Text content is substantial (likely a complete answer)
            if text_content and len(text_content.strip()) > 100:
                # Additional heuristics for final responses
                final_indicators = [
                    'currently', 'as of', 'recent', 'latest', 'according to',
                    'sources indicate', 'reports suggest', 'confirmed',
                    'breaking', 'update', 'announced'
                ]
                
                text_lower = text_content.lower()
                if any(indicator in text_lower for indicator in final_indicators):
                    return True
                
                # If it's a substantial response (>200 chars), likely final
                if len(text_content.strip()) > 200:
                    return True
            
            # Check 3: No function calls present (indicates completion)
            if hasattr(event, 'get_function_calls') and callable(event.get_function_calls):
                if not event.get_function_calls():
                    return len(text_content.strip()) > 50  # Has content but no more function calls
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking if final response: {e}")
            # If text is substantial, assume it's final
            return len(text_content.strip()) > 150

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        """Executes the agent's logic based on the incoming A2A request."""
        logger.info(f"🚀 Starting execution for task: {context.task_id}")
        
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            updater.submit()
        updater.start_work()
        
        # Log the incoming message
        message_text = ""
        for part in context.message.parts:
            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                message_text += part.root.text + " "
        logger.info(f"🚀 Processing message: {message_text[:200]}...")
        
        try:
            await self._process_request(
                genai_types.UserContent(
                    parts=convert_a2a_parts_to_genai(context.message.parts),
                ),
                context.context_id,
                updater,
            )
            
            logger.info(f"✅ Execution completed for task: {context.task_id}")
            
        except Exception as e:
            logger.error(f"❌ Error during execution: {e}")
            # Provide fallback response
            error_response = [TextPart(text=f"Search request processed, but encountered an issue: {str(e)}")]
            updater.add_artifact(error_response)
            updater.complete()

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
            return genai_types.Part(
                inline_data=genai_types.Blob(
                    data=part.file.bytes, mime_type=part.file.mime_type
                )
            )
        raise ValueError(f'Unsupported file type: {type(part.file)}')
    raise ValueError(f'Unsupported part type: {type(part)}')


def convert_genai_parts_to_a2a(parts: list[genai_types.Part]) -> list[Part]:
    """Converts a list of Google GenAI Part objects to a list of A2A Part objects."""
    converted_parts = []
    for part in parts:
        if part.text or part.file_data or part.inline_data:
            try:
                converted_part = convert_genai_part_to_a2a(part)
                converted_parts.append(converted_part)
            except Exception as e:
                logger.warning(f"Failed to convert part: {e}")
                if part.text:
                    converted_parts.append(TextPart(text=part.text))
    return converted_parts


def convert_genai_part_to_a2a(part: genai_types.Part) -> Part:
    """Converts a single Google GenAI Part object to an A2A Part object."""
    if part.text:
        return TextPart(text=part.text)
    if part.file_data:
        return Part(
            root=FilePart(
                file=FileWithUri(
                    uri=part.file_data.file_uri,
                    mime_type=part.file_data.mime_type,
                )
            )
        )
    if part.inline_data:
        return Part(
            root=FilePart(
                file=FileWithBytes(
                    bytes=part.inline_data.data,
                    mime_type=part.inline_data.mime_type,
                )
            )
        )
    raise ValueError(f'Unsupported part type: {part}')