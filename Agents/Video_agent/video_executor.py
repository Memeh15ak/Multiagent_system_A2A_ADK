import asyncio
import logging
import os
import tempfile
import re
from collections.abc import AsyncIterable

from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Artifact, FilePart, FileWithBytes, Part, TaskState,
    TaskStatus, TextPart, UnsupportedOperationError
)
from a2a.utils.errors import ServerError

from Agents.Video_agent.video_adk import create_video_agent, video_manager 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ADKVideoAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs the ADK-based Video Analysis Agent."""

    def __init__(self):
        self._agent = create_video_agent
        self.runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        logger.info(f"Starting execution for task {context.task_id}")
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        updater.submit()
        updater.start_work()

        logger.info(f"Cleaning up previous VideoManager state for task {context.task_id}...")
        video_manager.cleanup()
        logger.info("VideoManager state cleaned.")

        temp_files_to_clean = []
        try:
            augmented_prompt, temp_files_to_clean = await self._prepare_message_and_files(context.message.parts)
            
            genai_content = genai_types.UserContent(parts=[genai_types.Part(text=augmented_prompt)])

            async for _ in self._process_request(genai_content, context.context_id, updater):
                pass  

        except Exception as e:
            logger.error(f"Error in execute for task {context.task_id}: {e}", exc_info=True)
            updater.failed(f"Execution failed: {str(e)}")
        finally:
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
        if temp_file_paths:
            path_info = "The user has provided video files located at the following paths: " + \
                        ", ".join(f"'{p}'" for p in temp_file_paths) + ". "
            return f"{path_info}\n\nUser request: {user_prompt}", temp_file_paths
        
        return user_prompt, temp_file_paths

    async def _process_request(
        self,
        new_message: genai_types.Content,
        session_id: str,
        task_updater: TaskUpdater,
    ) -> AsyncIterable[TaskStatus | Artifact]:
        """Runs the ADK agent and streams updates back to the A2A framework."""
        has_final_response = False
        await self._upsert_session(session_id)
        logger.info(f"Starting ADK agent processing for session {session_id}")

        async for event in self.runner.run_async(session_id=session_id, user_id='a2a_user', new_message=new_message):
            if event.is_final_response():
                logger.info(f"Received final response for session {session_id}.")
                response_parts = convert_genai_parts_to_a2a(event.content.parts)
                task_updater.add_artifact(response_parts or [TextPart(text="Video analysis completed.")])
                task_updater.complete()
                has_final_response = True
                yield TaskStatus(state=TaskState.completed)
                break
            
            elif event.get_function_calls() or (event.content and event.content.parts):
                logger.info("Sending interim update (agent is thinking or calling a tool).")
                content_parts = convert_genai_parts_to_a2a(event.content.parts)
                if content_parts:
                    task_updater.update_status(
                        TaskState.working, message=task_updater.new_agent_message(content_parts)
                    )

        if not has_final_response:
            logger.warning(f"Session {session_id} ended without a final response.")
            task_updater.failed("Task did not complete successfully.")
            yield TaskStatus(state=TaskState.failed)

    async def _upsert_session(self, session_id: str):
        """Creates a session if it doesn't exist."""
        session = await self.runner.session_service.get_session(app_name=self.runner.app_name, user_id='a2a_user', session_id=session_id)
        if not session:
            session = await self.runner.session_service.create_session(app_name=self.runner.app_name, user_id='a2a_user', session_id=session_id)
        return session

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        logger.warning(f"Cancellation requested for task {context.task_id}, but not supported.")
        raise ServerError(error=UnsupportedOperationError())

def convert_genai_parts_to_a2a(parts: list[genai_types.Part] | None) -> list[Part]:
    """Converts a list of Google GenAI Part objects to a list of A2A Part objects."""
    if not parts:
        return []
    return [Part(root=TextPart(text=p.text)) for p in parts if p.text]