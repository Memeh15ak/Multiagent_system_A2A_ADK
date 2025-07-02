#!/usr/bin/env python3
"""
Simplified CLI client for multimedia file processing with agent executor integration.
Only requires --query and --multimedia_file arguments.
"""

import asyncio
import argparse
import httpx
import logging
import os
from uuid import uuid4
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

def get_multimedia_mime_type(file_path: str) -> str:
    """Get MIME type for any multimedia file"""
    ext = Path(file_path).suffix.lower()
    mime_types = {
        # Audio
        '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.flac': 'audio/flac', 
        '.ogg': 'audio/ogg', '.m4a': 'audio/mp4', '.aac': 'audio/aac',
        # Images  
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
        '.gif': 'image/gif', '.bmp': 'image/bmp', '.webp': 'image/webp',
        # Video
        '.mp4': 'video/mp4', '.avi': 'video/x-msvideo', '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska', '.webm': 'video/webm',
        #excel
        '.csv': 'text/csv',
        '.xlsx':'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    }
    return mime_types.get(ext, 'application/octet-stream')

def validate_multimedia_file(file_path: str) -> bool:
    """Validate that the file exists and is a supported multimedia format"""
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return False
    
    ext = Path(file_path).suffix.lower()
    supported = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.jpg', '.jpeg', 
                '.png', '.gif', '.bmp', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.webm','csv','xlsx']
    return ext in supported

async def create_message_with_multimedia_path(query: str, multimedia_file_path: str) -> dict:
    """Create message payload with multimedia file path"""
    parts = []
    
    # Add text query
    if query.strip():
        text_part = TextPart(text=query)
        parts.append(Part(root=text_part))
    
    # Add multimedia file
    if multimedia_file_path and validate_multimedia_file(multimedia_file_path):
        mime_type = get_multimedia_mime_type(multimedia_file_path)
        abs_path = os.path.abspath(multimedia_file_path)
        
        file_with_uri = FileWithUri(uri=abs_path, mime_type=mime_type)
        file_part = FilePart(file=file_with_uri)
        parts.append(Part(root=file_part))
        
        logger.info(f"Added {mime_type} file: {abs_path}")
    
    message = Message(role='user', parts=parts, messageId=uuid4().hex)
    return {'message': message}

async def poll_task_until_completion(client: A2AClient, task_id: str, max_attempts: int = 150):
    """Polls the task status until completion"""
    for attempt in range(max_attempts):
        try:
            query_params = TaskQueryParams(id=task_id)
            task_response = await client.get_task(GetTaskRequest(params=query_params))
            
            if isinstance(task_response.root, GetTaskSuccessResponse):
                task = task_response.root.result
                if task.status.state in [TaskState.completed, TaskState.failed, TaskState.canceled]:
                    return task
                    
        except Exception as e:
            logger.warning(f"Polling error: {e}")
        
        await asyncio.sleep(2)
    
    logger.error("Task polling timed out")
    return None

def extract_text_from_task(task: Task) -> str:
    """Extract all text content from a Task object"""
    result_texts = []
    
    # Check artifacts
    if hasattr(task, 'artifacts') and task.artifacts:
        for artifact in task.artifacts:
            if hasattr(artifact, 'parts') and artifact.parts:
                for part in artifact.parts:
                    if hasattr(part, 'root') and isinstance(part.root, TextPart):
                        if part.root.text and part.root.text.strip():
                            result_texts.append(part.root.text.strip())
    
    # Check status message
    if hasattr(task, 'status') and task.status and hasattr(task.status, 'message'):
        if task.status.message and hasattr(task.status.message, 'parts'):
            for part in task.status.message.parts:
                if hasattr(part, 'root') and isinstance(part.root, TextPart):
                    if part.root.text and part.root.text.strip():
                        result_texts.append(part.root.text.strip())
    
    return "\n\n".join(result_texts)

def print_task_results(task: Task):
    """Extract and print results from completed task"""
    if not task:
        print("No task results to display.")
        return
    
    print(f"\n--- Task Results (Status: {task.status.state}) ---")
    
    result_text = extract_text_from_task(task)
    
    if result_text.strip():
        print(result_text)
        print("-" * 50)
    else:
        print("No readable text content found in the completed task.")
    
    if task.status.state == TaskState.failed:
        print("\n--- Task Failed ---")
        if hasattr(task.status, 'error') and task.status.error:
            print(f"Error: {task.status.error.message}")

async def main():
    parser = argparse.ArgumentParser(description="Simplified multimedia processing client")
    parser.add_argument("--query", type=str, required=True, help="Your query/request")
    parser.add_argument("--multimedia_file", type=str, help="Path to multimedia file (optional)")
    
    args = parser.parse_args()

    # Validate file only if provided
    if args.multimedia_file and not validate_multimedia_file(args.multimedia_file):
        print(f"Error: Invalid or unsupported multimedia file: {args.multimedia_file}")
        return

    # Setup client
    timeout_config = httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0)
    
    async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
        client = A2AClient(url="http://localhost:10020", httpx_client=httpx_client)

        try:
            # Create message with or without multimedia file
            if args.multimedia_file:
                message_data = await create_message_with_multimedia_path(args.query, args.multimedia_file)
            else:
                # Create text-only message for queries like report generation
                text_part = TextPart(text=args.query)
                message = Message(role='user', parts=[Part(root=text_part)], messageId=uuid4().hex)
                message_data = {'message': message}
            
            send_message_payload = {
                **message_data,
                'agentId': "research_orchestrator",
                'userId': "cli_user",
            }
            
            request = SendMessageRequest(params=MessageSendParams(**send_message_payload))
            response = await client.send_message(request)
            
            # Handle response
            if isinstance(response.root, SendMessageSuccessResponse):
                if isinstance(response.root.result, Task):
                    task_id = response.root.result.id
                    logger.info(f"Task created: {task_id}. Polling for completion...")
                    completed_task = await poll_task_until_completion(client, task_id)
                    print_task_results(completed_task)
                elif isinstance(response.root.result, Message):
                    print("\n--- Direct Response ---")
                    for part_item in response.root.result.parts:
                        if hasattr(part_item, 'root') and isinstance(part_item.root, TextPart):
                            print(part_item.root.text)
            
            elif isinstance(response.root, JSONRPCErrorResponse):
                print(f"Error: {response.root.error.message}")

        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())