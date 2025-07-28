import os
import time
import logging
import tempfile
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections.abc import AsyncIterable

# Correct imports
try:
    from docx import Document  # This should work with python-docx
except ImportError:
    print("Warning: python-docx not installed. Install with: pip install python-docx")
    Document = None

import google.generativeai as genai
from google.adk import Runner
from google.adk.agents import Agent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.server.tasks.task_updater import Message  # Add this import for proper error handling
from a2a.types import (
    Artifact, FilePart, FileWithBytes, Part, TaskState,
    TaskStatus, TextPart, UnsupportedOperationError
)
from a2a.utils.errors import ServerError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FIXED MODEL NAMES - Remove problematic prefixes/suffixes
MODEL_FOR_ADK_AGENT = "gemini-1.5-flash"  # No "-latest" suffix
MODEL_FOR_ANALYSIS = "gemini-1.5-flash"   # Use consistent model or try "gemini-2.0-flash-exp"

def extract_text_from_docx(path: str) -> str | None:
    """Opens and reads a .docx file, returning its text content."""
    if Document is None:
        logger.error("python-docx not installed. Cannot read .docx files.")
        return None
    
    try:
        document = Document(path)
        return '\n'.join([para.text for para in document.paragraphs])
    except Exception as e:
        logger.warning(f"Could not read docx file {os.path.basename(path)}: {e}")
        return None

def extract_text_from_txt(path: str) -> str | None:
    """Opens and reads a .txt file, returning its text content."""
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            logger.warning(f"Could not read txt file {os.path.basename(path)}: {e}")
            return None
    except Exception as e:
        logger.warning(f"Could not read txt file {os.path.basename(path)}: {e}")
        return None

def analyze_multiple_files(file_paths: list[str], query: str) -> dict:
    """
    Analyzes content from a list of files to answer a user's query.
    Handles .docx and .txt files by extracting text, and uploads other supported file types.
    """
    logger.info(f"--- Tool Activated: analyze_multiple_files ---")
    logger.info(f"  Query: {query}")
    logger.info(f"  Files: {file_paths}")

    if not file_paths:
        return {"status": "error", "message": "No files provided for analysis."}

    prompt_parts = []
    files_to_delete = []

    for path in file_paths:
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}. Skipping.")
            continue

        file_basename = os.path.basename(path)
        file_extension = Path(path).suffix.lower()
        
        # Handle text-based files by extracting content
        if file_extension == '.docx':
            logger.info(f"Extracting text from DOCX: {file_basename}...")
            text_content = extract_text_from_docx(path)
            if text_content:
                formatted_text = f"--- Content from {file_basename} ---\n{text_content}\n--- End of {file_basename} ---"
                prompt_parts.append(formatted_text)
            else:
                logger.warning(f"Could not extract text from DOCX: {file_basename}")
        
        elif file_extension in ['.txt', '.md', '.py', '.js', '.json', '.xml', '.csv']:
            logger.info(f"Extracting text from {file_extension.upper()}: {file_basename}...")
            text_content = extract_text_from_txt(path)
            if text_content:
                formatted_text = f"--- Content from {file_basename} ---\n{text_content}\n--- End of {file_basename} ---"
                prompt_parts.append(formatted_text)
            else:
                logger.warning(f"Could not extract text from {file_extension.upper()}: {file_basename}")
        
        else:
            # Upload binary files (PDF, images, etc.)
            logger.info(f"Uploading binary file: {file_basename}...")
            try:
                uploaded_file = genai.upload_file(path=path)
                prompt_parts.append(uploaded_file)
                files_to_delete.append(uploaded_file)
            except Exception as e:
                logger.warning(f"Could not upload '{file_basename}'. It might be an unsupported file type. Error: {e}")

    if not prompt_parts:
        return {"status": "error", "message": "No valid or readable files were provided."}

    # Wait for uploaded files to be processed
    for file in files_to_delete:
        while file.state.name == "PROCESSING":
            logger.info(f"Waiting for {file.display_name}...")
            time.sleep(3)
            file = genai.get_file(name=file.name)
        
        if file.state.name == "FAILED":
            logger.error(f"File processing failed for {file.display_name}")
            # Clean up any successfully uploaded files
            for f in files_to_delete: 
                try:
                    genai.delete_file(f.name)
                except:
                    pass
            return {"status": "error", "message": f"File processing failed for {file.display_name}"}

    logger.info("All content ready. Asking Gemini for analysis...")
    try:
        system_instruction = """You are a meticulous document analyst. Your task is to answer the user's query based ONLY on the provided documents.

        **CRITICAL RULES:**
        1. Treat each document as a completely separate and distinct source.
        2. **DO NOT** mix, merge, or blend facts between different documents unless explicitly asked to compare.
        3. If information comes from a specific file, you must cite that file in your answer. For example, "According to document.pdf..." or "Based on the content from report.txt...".
        4. If the answer to a part of the query is not in any document, you must state that explicitly.
        5. Provide detailed, comprehensive answers when possible.
        6. When comparing documents, clearly distinguish between sources.
        7. If asked for summaries, provide structured, well-organized responses.
        
        **Response Format:**
        - Use clear headings and structure when appropriate
        - Cite sources for each piece of information
        - If multiple documents are involved, organize your response by document or by topic as appropriate
        """

        final_prompt = [
            system_instruction,
            "--- USER QUERY ---",
            query,
            "--- PROVIDED DOCUMENTS ---"
        ] + prompt_parts
        
        model = genai.GenerativeModel(MODEL_FOR_ANALYSIS)
        response = model.generate_content(final_prompt)
        
        if response.text:
            return {"status": "success", "analysis": response.text.strip()}
        else:
            return {"status": "error", "message": "No response generated from the model."}
            
    except Exception as e:
        logger.error(f"Error during Gemini content generation: {e}", exc_info=True)
        return {"status": "error", "message": f"Analysis failed: {str(e)}"}
    finally:
        # Clean up uploaded files
        if files_to_delete:
            logger.info("Cleaning up uploaded files...")
            for file in files_to_delete:
                try:
                    genai.delete_file(name=file.name)
                    logger.info(f"Deleted uploaded file: {file.display_name}")
                except Exception as e:
                    logger.warning(f"Could not delete file {file.display_name}: {e}")

# Memory context for the agent
memory = [
    "The user previously asked about 'project_update.docx' for its key points.",
    "The last file analyzed was 'Q4_report.pdf'. It contained sales figures and challenges.",
    "User prefers summaries to be in bullet points.",
    "User tends to ask follow-up questions about comparative analysis.",
    "The 2023 budget file 'budget_2023.xlsx' showed an allocated marketing budget of $50,000.",
]

# FIXED AGENT CREATION - Remove the "gemini/" prefix
create_file_rag_agent = Agent(
    name="file_rag_gemini_router",
    model=MODEL_FOR_ADK_AGENT,  # Just use the model name directly
    description="Agent that analyzes one or more files by calling the appropriate tool.",
    instruction=(
        f"""You are an expert file analysis assistant. Your purpose is to provide accurate answers to user questions by analyzing the content of local files using your tools.

        **Core Principle:** You must base your answers **exclusively** on the output provided by the `analyze_multiple_files` tool. If the tool's analysis does not contain the answer, you must state that the information is not available in the provided documents.

        **Tool Usage Rules:**
        1. When a user's query contains file paths or file names, you MUST call the `analyze_multiple_files` tool.
        2. Extract all file paths from the user's query and pass them as a list to the `file_paths` parameter.
        3. Create a clear and specific `query` parameter that reflects the user's intent.
        4. If the user refers to previously analyzed files from memory context, include those files in your analysis.
        5. Handle errors gracefully and provide helpful feedback to the user.

        **Supported File Types:**
        - Text files: .txt, .md, .py, .js, .json, .xml, .csv
        - Word documents: .docx
        - PDFs and other binary files (uploaded to Gemini)

        **Response Guidelines:**
        - Provide comprehensive, well-structured answers
        - Always cite sources when referencing specific documents
        - If comparing multiple files, clearly distinguish between sources
        - Use formatting (headings, bullet points) to improve readability
        - If a file cannot be processed, explain why and suggest alternatives

        **Memory Context:** {memory}
        """
    ),
    tools=[analyze_multiple_files], 
)

class ADKFileRagExecutor(AgentExecutor):
    """An AgentExecutor that runs the ADK-based Multi-File RAG Agent."""

    def __init__(self):
        self._agent = create_file_rag_agent
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

        temp_files_to_clean = []
        try:
            augmented_prompt, temp_files_to_clean = await self._prepare_message_and_files(context.message.parts)
            
            logger.info(f"Augmented prompt for ADK agent: {augmented_prompt[:500]}...")
            genai_content = genai_types.UserContent(parts=[genai_types.Part(text=augmented_prompt)])

            async for _ in self._process_request(genai_content, context.context_id, updater):
                pass

        except Exception as e:
            logger.error(f"Error in execute for task {context.task_id}: {e}", exc_info=True)
            # FIXED ERROR HANDLING - Use proper Message object
            try:
                error_message = Message(content=f"Execution failed: {str(e)}")
                updater.failed(error_message)
            except Exception as msg_error:
                logger.error(f"Could not create Message object: {msg_error}")
                error_parts = [TextPart(text=f"Execution failed: {str(e)}")]
                updater.failed(updater.new_agent_message(error_parts))
        finally:
            # Clean up temporary files
            for path in temp_files_to_clean:
                try:
                    if os.path.exists(path):
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
                # Determine file extension
                original_filename = part_root.file.filename or 'unknown_file'
                file_suffix = os.path.splitext(original_filename)[1] or '.tmp'
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
                    tmp_file.write(part_root.file.bytes)
                    temp_file_paths.append(tmp_file.name)
                    logger.info(f"Saved incoming file '{original_filename}' to temporary path: {tmp_file.name}")
                    
            elif isinstance(part_root, TextPart):
                text_parts.append(part_root.text)

        user_query = " ".join(text_parts)
        
        if temp_file_paths:
            file_paths_str = "', '".join(temp_file_paths)
            augmented_prompt = (
                f"The user has provided files located at the following local paths: ['{file_paths_str}']. "
                f"Please analyze them to answer the user's query.\n\n"
                f"User's Query: {user_query}"
            )
            return augmented_prompt, temp_file_paths
        
        return user_query, temp_file_paths

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

        try:
            async for event in self.runner.run_async(session_id=session_id, user_id='a2a_user', new_message=new_message):
                if event.is_final_response():
                    logger.info(f"Received final response for session {session_id}")
                    response_parts = convert_genai_parts_to_a2a(event.content.parts)
                    task_updater.add_artifact(response_parts or [TextPart(text="File analysis completed.")])
                    task_updater.complete()
                    has_final_response = True
                    yield TaskStatus(state=TaskState.completed)
                    break
                
                elif event.get_function_calls() or (event.content and event.content.parts):
                    logger.info("Sending interim update (agent is thinking or calling a tool)")
                    content_parts = convert_genai_parts_to_a2a(event.content.parts)
                    if content_parts:
                        task_updater.update_status(
                            TaskState.working, 
                            message=task_updater.new_agent_message(content_parts)
                        )

        except Exception as e:
            logger.error(f"Error during agent processing: {e}", exc_info=True)
            # FIXED ERROR HANDLING - Use proper Message object
            try:
                error_message = Message(content=f"Agent processing failed: {str(e)}")
                task_updater.failed(error_message)
            except Exception as msg_error:
                logger.error(f"Could not create Message object: {msg_error}")
                error_parts = [TextPart(text=f"Agent processing failed: {str(e)}")]
                task_updater.failed(task_updater.new_agent_message(error_parts))
            yield TaskStatus(state=TaskState.failed)
            return

        if not has_final_response:
            logger.warning(f"Session {session_id} ended without a final response")
            # FIXED ERROR HANDLING
            try:
                error_message = Message(content="Task did not complete successfully.")
                task_updater.failed(error_message)
            except Exception:
                error_parts = [TextPart(text="Task did not complete successfully.")]
                task_updater.failed(task_updater.new_agent_message(error_parts))
            yield TaskStatus(state=TaskState.failed)

    async def _upsert_session(self, session_id: str):
        """Creates a session if it doesn't exist."""
        try:
            session = await self.runner.session_service.get_session(
                app_name=self.runner.app_name, 
                user_id='a2a_user', 
                session_id=session_id
            )
            if not session:
                await self.runner.session_service.create_session(
                    app_name=self.runner.app_name, 
                    user_id='a2a_user', 
                    session_id=session_id
                )
        except Exception as e:
            logger.error(f"Error creating/accessing session {session_id}: {e}")
            raise

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        logger.warning(f"Cancellation requested for task {context.task_id}, but not supported.")
        raise ServerError(error=UnsupportedOperationError())

def convert_genai_parts_to_a2a(parts: list[genai_types.Part] | None) -> list[Part]:
    """Converts a list of Google GenAI Part objects to a list of A2A Part objects."""
    if not parts:
        return []
    
    result = []
    for part in parts:
        if hasattr(part, 'text') and part.text:
            result.append(Part(root=TextPart(text=part.text)))
    
    return result

# Optional: Add debugging function to check available models
def debug_available_models():
    """Debug function to check what models are available in the registry."""
    try:
        from google.adk.models.registry import LLMRegistry
        available_models = LLMRegistry.list_models()
        logger.info(f"Available models: {available_models}")
        return available_models
    except Exception as e:
        logger.error(f"Could not check available models: {e}")
        return []

# Uncomment the line below to debug model availability
# debug_available_models()