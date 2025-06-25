#!/usr/bin/env python3
"""
Enhanced CLI client for audio file processing with agent executor integration.
Supports direct file path passing for audio files to translation tools.
"""

import asyncio
import argparse
import httpx
import logging
import os
import mimetypes
from uuid import uuid4
from typing import Any, Optional, Tuple, List
from pathlib import Path
from a2a.client import A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendMessageSuccessResponse,
    JSONRPCErrorResponse,
    Task,
    Message,
    TextPart,
    FilePart,
    FileWithUri,
    Part,
    TaskState,
    GetTaskRequest,
    GetTaskSuccessResponse,
    TaskQueryParams,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def detect_audio_file_path(query: str) -> Tuple[str, Optional[str]]:
    """
    Detect if query contains an audio file path and extract it
    Returns: (cleaned_query, file_path)
    """
    # Common audio file extensions
    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus', '.aiff', '.au']
    
    words = query.split()
    file_path = None
    cleaned_words = []
    
    for word in words:
        # Check if word looks like a file path with audio extension
        if any(word.lower().endswith(ext) for ext in audio_extensions):
            # Check if it's a valid file path
            if os.path.exists(word):
                file_path = word
                logger.info(f"Detected audio file path: {file_path}")
            else:
                # Keep the word if file doesn't exist, might be part of query
                cleaned_words.append(word)
        else:
            cleaned_words.append(word)
    
    cleaned_query = ' '.join(cleaned_words)
    logger.info(f"Cleaned query and audio file path: '{cleaned_query}', '{file_path}'")
    return cleaned_query, file_path

def detect_multimedia_file_path(query: str) -> Tuple[str, Optional[str]]:
    """
    Detect if query contains any multimedia file path (audio, image, video) and extract it
    Returns: (cleaned_query, file_path)
    """
    # Comprehensive multimedia file extensions
    multimedia_extensions = [
        # Audio
        '.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus', '.aiff', '.au',
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.svg', '.ico',
        # Video
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'
    ]
    
    words = query.split()
    file_path = None
    cleaned_words = []
    
    for word in words:
        # Check if word looks like a file path with multimedia extension
        if any(word.lower().endswith(ext) for ext in multimedia_extensions):
            # Check if it's a valid file path
            if os.path.exists(word):
                file_path = word
                logger.info(f"Detected multimedia file path: {file_path}")
            else:
                # Keep the word if file doesn't exist, might be part of query
                cleaned_words.append(word)
        else:
            cleaned_words.append(word)
    
    cleaned_query = ' '.join(cleaned_words)
    logger.info(f"Cleaned query and multimedia file path: '{cleaned_query}', '{file_path}'")
    return cleaned_query, file_path

def get_audio_mime_type(file_path: str) -> str:
    """Get MIME type for audio file"""
    # First try Python's mimetypes module
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type and mime_type.strip() and mime_type.startswith('audio/'):
        logger.info(f"Detected MIME type from mimetypes: {mime_type}")
        return mime_type.strip()
    
    # Fallback to extension-based mapping for audio files
    ext = Path(file_path).suffix.lower()
    audio_mime_types = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.m4a': 'audio/mp4',
        '.aac': 'audio/aac',
        '.wma': 'audio/x-ms-wma',
        '.opus': 'audio/opus',
        '.aiff': 'audio/aiff',
        '.au': 'audio/basic'
    }
    
    fallback_mime = audio_mime_types.get(ext)
    if fallback_mime:
        logger.info(f"Using fallback MIME type for {ext}: {fallback_mime}")
        return fallback_mime
    
    # Default to MP3 for unknown audio types
    logger.warning(f"Could not determine MIME type for {ext}, using default audio/mpeg")
    return 'audio/mpeg'

def get_multimedia_mime_type(file_path: str) -> str:
    """Get MIME type for any multimedia file"""
    # First try Python's mimetypes module
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type and mime_type.strip():
        logger.info(f"Detected MIME type from mimetypes: {mime_type}")
        return mime_type.strip()
    
    # Fallback to extension-based mapping
    ext = Path(file_path).suffix.lower()
    mime_types = {
        # Audio
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.m4a': 'audio/mp4',
        '.aac': 'audio/aac',
        '.wma': 'audio/x-ms-wma',
        '.opus': 'audio/opus',
        '.aiff': 'audio/aiff',
        '.au': 'audio/basic',
        # Images
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.svg': 'image/svg+xml',
        '.ico': 'image/x-icon',
        # Video
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.wmv': 'video/x-ms-wmv',
        '.flv': 'video/x-flv',
        '.webm': 'video/webm',
        '.m4v': 'video/x-m4v'
    }
    
    fallback_mime = mime_types.get(ext)
    if fallback_mime:
        logger.info(f"Using fallback MIME type for {ext}: {fallback_mime}")
        return fallback_mime
    
    # Default based on file type
    if ext in ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']:
        return 'audio/mpeg'
    elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        return 'image/jpeg'
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        return 'video/mp4'
    
    logger.warning(f"Could not determine MIME type for {ext}, using default application/octet-stream")
    return 'application/octet-stream'

def validate_audio_file(file_path: str) -> bool:
    """Validate that the file exists and is a supported audio format"""
    if not os.path.exists(file_path):
        logger.error(f"Audio file does not exist: {file_path}")
        return False
    
    if not os.path.isfile(file_path):
        logger.error(f"Path is not a file: {file_path}")
        return False
    
    # Check if it's a supported audio format
    ext = Path(file_path).suffix.lower()
    supported_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus', '.aiff', '.au']
    
    if ext not in supported_extensions:
        logger.error(f"Unsupported audio format: {ext}")
        return False
    
    file_size = os.path.getsize(file_path)
    logger.info(f"Audio file validation passed: {file_path} ({file_size} bytes)")
    return True

def validate_multimedia_file(file_path: str) -> bool:
    """Validate that the file exists and is a supported multimedia format"""
    if not os.path.exists(file_path):
        logger.error(f"Multimedia file does not exist: {file_path}")
        return False
    
    if not os.path.isfile(file_path):
        logger.error(f"Path is not a file: {file_path}")
        return False
    
    # Check if it's a supported multimedia format
    ext = Path(file_path).suffix.lower()
    supported_extensions = [
        # Audio
        '.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus', '.aiff', '.au',
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif',
        # Video
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'
    ]
    
    if ext not in supported_extensions:
        logger.error(f"Unsupported multimedia format: {ext}")
        return False
    
    file_size = os.path.getsize(file_path)
    logger.info(f"Multimedia file validation passed: {file_path} ({file_size} bytes)")
    return True

async def create_message_with_audio_path(query: str, audio_file_path: str) -> dict[str, Any]:
    """Create message payload with audio file path for translation processing"""
    parts = []
    
    # Add text part with enhanced context for audio translation
    if query.strip():
        # Enhance the query with audio processing context
        enhanced_query = f"""AUDIO TRANSLATION REQUEST:
User Query: {query}
Audio File: {audio_file_path}

Please process this audio file for translation. The file should be:
1. Loaded and analyzed for speech content
2. Transcribed to text
3. Translated if requested
4. Provide both transcription and translation results

Route this to the appropriate audio processing agent for translation."""
        
        text_part = TextPart(text=enhanced_query)
        parts.append(Part(root=text_part))
    
    # Add audio file path as FileWithUri
    if audio_file_path and validate_audio_file(audio_file_path):
        try:
            # Get MIME type
            mime_type = get_audio_mime_type(audio_file_path)
            logger.info(f"Using MIME type: '{mime_type}' for audio file: {audio_file_path}")
            
            # Convert to absolute path if it's relative
            if not os.path.isabs(audio_file_path):
                abs_path = os.path.abspath(audio_file_path)
            else:
                abs_path = audio_file_path
            
            # Normalize path separators for consistency
            normalized_path = os.path.normpath(abs_path)
            
            logger.info(f"Using normalized audio path: {normalized_path}")
            
            # Create FileWithUri object with just the path
            file_with_uri = FileWithUri(
                uri=normalized_path,
                mime_type=mime_type
            )
            
            # Create FilePart object
            file_part = FilePart(file=file_with_uri)
            parts.append(Part(root=file_part))
            
            logger.info(f"Successfully added audio file path for translation: {normalized_path}")
            
        except Exception as e:
            logger.error(f"Error processing audio file path {audio_file_path}: {e}")
            # Add error message as text
            error_text = TextPart(text=f"Error: Could not process audio file {audio_file_path}: {e}")
            parts.append(Part(root=error_text))
    else:
        if audio_file_path:
            # Add error message if validation failed
            error_text = TextPart(text=f"Error: Audio file validation failed for {audio_file_path}")
            parts.append(Part(root=error_text))
    
    # Create the message structure
    message = Message(
        role='user',
        parts=parts,
        messageId=uuid4().hex
    )
    
    return {
        'message': message
    }

async def create_message_with_multimedia_path(query: str, multimedia_file_path: str) -> dict[str, Any]:
    """Create message payload with any multimedia file path"""
    parts = []
    
    # Determine file type for enhanced routing
    ext = Path(multimedia_file_path).suffix.lower()
    file_type = "unknown"
    if ext in ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus', '.aiff', '.au']:
        file_type = "audio"
    elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif']:
        file_type = "image"
    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']:
        file_type = "video"
    
    # Add text part with enhanced context for multimedia processing
    if query.strip():
        enhanced_query = f"""MULTIMEDIA PROCESSING REQUEST:
User Query: {query}
File: {multimedia_file_path}
File Type: {file_type.upper()}

Please process this {file_type} file according to the user's request. 
Route to the appropriate agent based on file type:
- Audio files: Route to audio processing/translation agent
- Image files: Route to image modification agent  
- Video files: Route to video processing agent

The file is available at the provided path for direct access."""
        
        text_part = TextPart(text=enhanced_query)
        parts.append(Part(root=text_part))
    
    # Add multimedia file path as FileWithUri
    if multimedia_file_path and validate_multimedia_file(multimedia_file_path):
        try:
            # Get MIME type
            mime_type = get_multimedia_mime_type(multimedia_file_path)
            logger.info(f"Using MIME type: '{mime_type}' for {file_type} file: {multimedia_file_path}")
            
            # Convert to absolute path if it's relative
            if not os.path.isabs(multimedia_file_path):
                abs_path = os.path.abspath(multimedia_file_path)
            else:
                abs_path = multimedia_file_path
            
            # Normalize path separators for consistency
            normalized_path = os.path.normpath(abs_path)
            
            logger.info(f"Using normalized {file_type} path: {normalized_path}")
            
            # Create FileWithUri object with the path
            file_with_uri = FileWithUri(
                uri=normalized_path,
                mime_type=mime_type
            )
            
            # Create FilePart object
            file_part = FilePart(file=file_with_uri)
            parts.append(Part(root=file_part))
            
            logger.info(f"Successfully added {file_type} file path: {normalized_path}")
            
        except Exception as e:
            logger.error(f"Error processing {file_type} file path {multimedia_file_path}: {e}")
            # Add error message as text
            error_text = TextPart(text=f"Error: Could not process {file_type} file {multimedia_file_path}: {e}")
            parts.append(Part(root=error_text))
    else:
        if multimedia_file_path:
            # Add error message if validation failed
            error_text = TextPart(text=f"Error: {file_type.capitalize()} file validation failed for {multimedia_file_path}")
            parts.append(Part(root=error_text))
    
    # Create the message structure
    message = Message(
        role='user',
        parts=parts,
        messageId=uuid4().hex
    )
    
    return {
        'message': message
    }

async def poll_task_until_completion(client: A2AClient, task_id: str, max_attempts: int = 150) -> Task | None:
    """Polls the task status until it's completed or failed with timeout protection"""
    attempts = 0
    
    while attempts < max_attempts:
        try:
            logger.info(f"Polling task: {task_id} (attempt {attempts + 1}/{max_attempts})")
            query_params = TaskQueryParams(id=task_id)
            task_response = await client.get_task(GetTaskRequest(params=query_params))
            
            if isinstance(task_response.root, JSONRPCErrorResponse):
                logger.error(f"Error getting task status: {task_response.root.error.message}")
                return None
            
            if isinstance(task_response.root, GetTaskSuccessResponse) and isinstance(task_response.root.result, Task):
                task = task_response.root.result
                logger.info(f"Task status: {task.status.state}")
                
                # Log interim messages
                if task.status.message and task.status.message.parts:
                    for part in task.status.message.parts:
                        if isinstance(part.root, TextPart):
                            logger.info(f"Agent interim message: {part.root.text}")

                if task.status.state in [TaskState.completed, TaskState.failed, TaskState.canceled]:
                    logger.info(f"Task {task.status.state}.")
                    return task
            else:
                logger.error(f"Unexpected response type when getting task: {task_response}")
                return None
                
        except Exception as e:
            logger.warning(f"Error during polling attempt {attempts + 1}: {e}")
        
        attempts += 1
        await asyncio.sleep(2)  # Poll every 2 seconds
    
    logger.error(f"Task polling timed out after {max_attempts} attempts")
    return None

def extract_text_from_task(task: Task) -> str:
    """Extract all text content from a Task object"""
    result_texts = []
    
    if not task:
        return ""
    
    logger.info("Extracting text from Task object...")
    
    # Check artifacts (primary location for results)
    if hasattr(task, 'artifacts') and task.artifacts:
        logger.info(f"Found {len(task.artifacts)} artifacts")
        for i, artifact in enumerate(task.artifacts):
            logger.info(f"Processing artifact {i}: {type(artifact)}")
            if hasattr(artifact, 'parts') and artifact.parts:
                for j, part in enumerate(artifact.parts):
                    logger.info(f"Processing artifact {i}, part {j}: {type(part)}")
                    if hasattr(part, 'root') and isinstance(part.root, TextPart):
                        text_content = part.root.text
                        if text_content and text_content.strip():
                            result_texts.append(text_content.strip())
                            logger.info(f"Extracted text content: {len(text_content)} characters")
    
    # Check status message (sometimes contains results)
    if hasattr(task, 'status') and task.status:
        if hasattr(task.status, 'message') and task.status.message:
            if hasattr(task.status.message, 'parts') and task.status.message.parts:
                logger.info("Checking status message for content...")
                for part in task.status.message.parts:
                    if hasattr(part, 'root') and isinstance(part.root, TextPart):
                        text_content = part.root.text
                        if text_content and text_content.strip():
                            result_texts.append(text_content.strip())
                            logger.info(f"Extracted text from status message: {len(text_content)} characters")
    
    # Join all text content
    full_text = "\n\n".join(result_texts)
    logger.info(f"Total extracted text: {len(full_text)} characters")
    
    return full_text

def print_task_results(task: Task):
    """Extract and print results from completed task"""
    if not task:
        print("No task results to display.")
        return
    
    print(f"\n--- Task Results (Status: {task.status.state}) ---")
    
    # Extract text content
    result_text = extract_text_from_task(task)
    
    if result_text.strip():
        print(result_text)
        print("-" * 50)
    else:
        print("No readable text content found in the completed task.")
    
    # Check for error information
    if task.status.state == TaskState.failed:
        print(f"\n--- Task Failed ---")
        if hasattr(task.status, 'error') and task.status.error:
            print(f"Error: {task.status.error.message} (Code: {task.status.error.code})")
        else:
            print("Task failed but no error details available")

async def main():
    parser = argparse.ArgumentParser(
        description="Enhanced CLI client for audio translation and multimedia processing."
    )
    parser.add_argument(
        "--query", 
        type=str, 
        required=True, 
        help="The query with optional file path (e.g., \"Translate this audio file /path/to/audio.mp3\")"
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        help="Explicit path to audio file for translation"
    )
    parser.add_argument(
        "--multimedia_file",
        type=str,
        help="Explicit path to any multimedia file (audio/image/video)"
    )
    parser.add_argument(
        "--agent_url", 
        type=str, 
        default="http://localhost:10020", 
        help="The URL of the target agent (default: orchestrator)"
    )
    parser.add_argument(
        "--agent_id",
        type=str,
        default="research_orchestrator",
        help="The agent ID to send the message to"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="HTTP timeout in seconds (default: 600 = 10 minutes)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--translation_mode",
        action="store_true",
        help="Enable special audio translation processing mode"
    )
    
    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Determine which file to process
    multimedia_file_path = None
    cleaned_query = args.query
    
    # Priority: explicit multimedia_file > explicit audio_file > detected in query
    if args.multimedia_file:
        multimedia_file_path = args.multimedia_file
        logger.info(f"Using explicit multimedia file: {multimedia_file_path}")
    elif args.audio_file:
        multimedia_file_path = args.audio_file
        logger.info(f"Using explicit audio file: {multimedia_file_path}")
    else:
        # Try to detect multimedia file in query
        cleaned_query, detected_file = detect_multimedia_file_path(args.query)
        multimedia_file_path = detected_file
        if multimedia_file_path:
            logger.info(f"Detected multimedia file in query: {multimedia_file_path}")
    
    # Set default query if only file provided
    if multimedia_file_path and not cleaned_query.strip():
        if args.translation_mode or args.audio_file:
            cleaned_query = "Please analyze and translate this audio file."
        else:
            cleaned_query = "Please analyze this multimedia file."
    
    # Validate multimedia file if provided
    if multimedia_file_path:
        logger.info(f"Processing with multimedia file: {multimedia_file_path}")
        logger.info(f"Query text: {cleaned_query}")
        
        if not validate_multimedia_file(multimedia_file_path):
            print(f"Error: Multimedia file validation failed for {multimedia_file_path}")
            return
    else:
        cleaned_query = args.query
        logger.info(f"Processing text-only query: {cleaned_query}")

    # Configure timeout
    timeout_config = httpx.Timeout(
        connect=30.0,
        read=args.timeout,
        write=30.0,
        pool=30.0
    )

    async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
        client = A2AClient(url=args.agent_url, httpx_client=httpx_client)

        logger.info(f"Sending to agent {args.agent_id} at {args.agent_url}...")
        logger.info(f"Using timeout: {args.timeout} seconds")

        try:
            # Create message with or without multimedia file path
            if multimedia_file_path:
                if args.translation_mode or args.audio_file:
                    # Use audio-specific processing
                    message_data = await create_message_with_audio_path(cleaned_query, multimedia_file_path)
                else:
                    # Use general multimedia processing
                    message_data = await create_message_with_multimedia_path(cleaned_query, multimedia_file_path)
            else:
                # Create text-only message
                text_part = TextPart(text=cleaned_query)
                message = Message(
                    role='user',
                    parts=[Part(root=text_part)],
                    messageId=uuid4().hex
                )
                message_data = {'message': message}
            
            # Add agent and user info
            send_message_payload = {
                **message_data,
                'agentId': args.agent_id,
                'userId': "cli_user",
            }
            
            request = SendMessageRequest(
                params=MessageSendParams(**send_message_payload)
            )

            response = await client.send_message(request)
            
            task_id_to_poll = None

            if isinstance(response.root, SendMessageSuccessResponse):
                if isinstance(response.root.result, Task):
                    task_id_to_poll = response.root.result.id
                    logger.info(f"Message sent. Task ID: {task_id_to_poll}. Polling for completion...")
                elif isinstance(response.root.result, Message):
                    logger.info("Received direct message response")
                    if response.root.result.parts:
                        print("\n--- Direct Agent Response ---")
                        for part_item in response.root.result.parts:
                            if hasattr(part_item, 'root') and isinstance(part_item.root, TextPart):
                                print(part_item.root.text)
                else:
                    logger.error(f"Unexpected result type: {type(response.root.result)}")

            elif isinstance(response.root, JSONRPCErrorResponse):
                logger.error(f"Agent returned error: {response.root.error.message}")
                print(f"Error: {response.root.error.message}")
            else:
                logger.error(f"Unknown response type: {response}")

            if task_id_to_poll:
                completed_task = await poll_task_until_completion(client, task_id_to_poll)
                print_task_results(completed_task)

        except httpx.ReadTimeout:
            logger.error(f"Request timed out after {args.timeout} seconds")
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=args.debug)
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())