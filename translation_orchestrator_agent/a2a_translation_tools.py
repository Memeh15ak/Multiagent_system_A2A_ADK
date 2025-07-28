# a2a_translation_tools.py - UPDATED VERSION WITH ALL AGENTS
import asyncio
import httpx
import logging
from typing import Dict, Any, Optional, List, Union
from uuid import uuid4
import time
import os
import json
from enum import Enum
from a2a.client import A2AClient
from a2a.types import (
    MessageSendParams, SendMessageRequest, SendMessageSuccessResponse,
    Task, Message, TextPart, TaskState, GetTaskRequest, 
    GetTaskSuccessResponse, TaskQueryParams,
)

logger = logging.getLogger(__name__)

# UPDATED Agent Configuration with all agents
AGENTS = {
    "web_search": {"url": "http://localhost:10012", "id": "web_searcher"},
    "code": {"url": "http://localhost:10011", "id": "code_executor"},
    "audio": {"url": "http://localhost:10013", "id": "audio_processor"},
    "image": {"url": "http://localhost:10014", "id": "img_to_img_processor"},
    "report": {"url": "http://localhost:10015", "id": "report_content_generator"},
    "excel": {"url": "http://localhost:10016", "id": "excel_analyzer"},
    "rag": {"url": "http://localhost:10019", "id": "rag_document_processor"},
    "image_analyzer": {"url": "http://localhost:10017", "id": "image_analysis_agent"},
    "image_generation": {"url": "http://localhost:10018", "id": "image_generation_agent"},
    "video": {"url": "http://localhost:10021", "id": "video_processor"}
}

REQUEST_TIMEOUT = 120.0
POLL_INTERVAL = 5.0
MAX_POLL_ATTEMPTS = 50

# CRITICAL FIX: Proper enum handling and serialization
def convert_to_serializable(obj):
    """Convert any object to a JSON-serializable format"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, 'model_dump'):
        # Pydantic models
        return convert_to_serializable(obj.model_dump())
    elif hasattr(obj, '__dict__'):
        # Regular objects
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    else:
        return str(obj)

def create_message_payload(message_text: str, agent_id: str, user_id: str = "orchestrator"):
    """Create a properly formatted message payload"""
    return {
        'message': {
            'role': 'user',  # Always use string, never enum
            'parts': [{'kind': 'text', 'text': message_text}],
            'messageId': uuid4().hex,
        },
        'agentId': agent_id,
        'userId': user_id,
    }

async def check_agent_availability(agent_url: str) -> bool:
    """Check if an agent is available"""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            response = await client.get(f"{agent_url}/health")
            return response.status_code == 200
    except Exception as e:
        logger.warning(f"Agent availability check failed: {e}")
        return False

async def poll_task_completion(client: A2AClient, task_id: str) -> Optional[Task]:
    """Poll task until completion"""
    for attempt in range(MAX_POLL_ATTEMPTS):
        try:
            query_params = TaskQueryParams(id=task_id)
            response = await client.get_task(GetTaskRequest(params=query_params))
            
            if isinstance(response.root, GetTaskSuccessResponse):
                task = response.root.result
                logger.info(f"Task {task_id} status: {task.status.state}")
                
                if task.status.state in [TaskState.completed, TaskState.failed, TaskState.canceled]:
                    return task
            
            await asyncio.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.warning(f"Polling error for task {task_id}: {e}")
    
    logger.error(f"Task {task_id} polling timed out after {MAX_POLL_ATTEMPTS} attempts")
    return None

def extract_text_from_task(task: Task) -> str:
    """Extract text content from a completed task"""
    if not task:
        return "No task result available"
    
    # Try to extract from artifacts first
    if hasattr(task, 'artifacts') and task.artifacts:
        for artifact in task.artifacts:
            if hasattr(artifact, 'parts') and artifact.parts:
                for part in artifact.parts:
                    if hasattr(part, 'text'):
                        return part.text
                    elif hasattr(part, 'root') and hasattr(part.root, 'text'):
                        return part.root.text
    
    # Try to extract from messages
    if hasattr(task, 'messages') and task.messages:
        for message in task.messages:
            if hasattr(message, 'parts') and message.parts:
                for part in message.parts:
                    if hasattr(part, 'text'):
                        return part.text
                    elif hasattr(part, 'root') and hasattr(part.root, 'text'):
                        return part.root.text
    
    # Fallback to status message
    if hasattr(task.status, 'message') and task.status.message:
        return task.status.message
    
    return f"Task completed with state: {task.status.state}"

def extract_text_from_message(message: Message) -> str:
    """Extract text content from a message"""
    if not message:
        return "No message content available"
    
    if hasattr(message, 'parts') and message.parts:
        for part in message.parts:
            if hasattr(part, 'text'):
                return part.text
            elif hasattr(part, 'root') and hasattr(part.root, 'text'):
                return part.root.text
            else:
                return message
    else:
        return message 


async def send_message_to_agent(
    agent_type: str, 
    message: str, 
    original_user_query: str,
    file_paths: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Send message to agent and return response - COMPLETELY FIXED VERSION"""
    
    if agent_type not in AGENTS:
        return {"status": "error", "error": f"Unknown agent: {agent_type}"}
    
    agent_config = AGENTS[agent_type]
    
    # Check agent availability
    if not await check_agent_availability(agent_config["url"]):
        return {"status": "error", "error": f"Agent {agent_type} at {agent_config['url']} is unavailable"}
    
    try:
        # Enhance message with file info
        enhanced_message = message
        if file_paths:
            enhanced_message += f"\n\n[FILES]\n" + "\n".join([f"Path: {p}" for p in file_paths])
        if file_urls:
            enhanced_message += f"\n\n[URLS]\n" + "\n".join([f"URL: {u}" for u in file_urls])
        
        # Create the payload with proper structure
        payload = create_message_payload(enhanced_message, agent_config["id"])
        
        # Convert to fully serializable format
        serializable_payload = convert_to_serializable(payload)
        
        # Log the payload for debugging
        logger.info(f"Sending to {agent_type} agent: {json.dumps(serializable_payload, indent=2)[:500]}...")
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as httpx_client:
            client = A2AClient(url=agent_config["url"], httpx_client=httpx_client)
            
            # CRITICAL: Create request with serializable data
            try:
                request = SendMessageRequest(params=MessageSendParams(**serializable_payload))
            except Exception as e:
                logger.error(f"Failed to create request: {e}")
                logger.error(f"Payload: {serializable_payload}")
                return {"status": "error", "error": f"Failed to create request: {e}"}
            
            # Send the request
            response = await client.send_message(request)
            logger.info(f"Raw response from {agent_type}: {response}")
            
            if isinstance(response.root, SendMessageSuccessResponse):
                result = response.root.result
                logger.info(f"Success response result type: {type(result)}")
                
                if isinstance(result, Task):
                    logger.info(f"Received Task {result.id} with state: {result.status.state}")
                    
                    # Poll for completion
                    completed_task = await poll_task_completion(client, result.id)
                    
                    if completed_task:
                        if completed_task.status.state == TaskState.completed:
                            response_text = extract_text_from_task(completed_task)
                            return {"status": "success", "result": response_text}
                        else:
                            error_msg = f"Task failed with state: {completed_task.status.state}"
                            if hasattr(completed_task.status, 'message') and completed_task.status.message:
                                error_msg += f" - {completed_task.status.message}"
                            return {"status": "error", "error": error_msg}
                    else:
                        return {"status": "error", "error": "Task polling timed out"}
                
                elif isinstance(result, Message):
                    logger.info(f"Received direct Message response")
                    response_text = extract_text_from_message(result)
                    return {"status": "success", "result": response_text}
                
                else:
                    logger.warning(f"Unexpected result type: {type(result)}")
                    return {"status": "error", "error": f"Unexpected result type: {type(result)}"}
            
            else:
                logger.error(f"Unexpected response type: {type(response.root)}")
                return {"status": "error", "error": f"Unexpected response format: {type(response.root)}"}
            
    except Exception as e:
        logger.error(f"Error sending message to {agent_type}: {e}", exc_info=True)
        return {"status": "error", "error": f"Communication error: {str(e)}"}

# Helper functions for file context
def get_file_paths_from_global_context() -> List[str]:
    """Get file paths from global context"""
    try:
        from translation_orchestrator_agent.adk_agent_executor import get_file_paths_from_context
        return get_file_paths_from_context()
    except ImportError:
        return []

def get_file_info_by_media_type(media_type: str) -> List[Dict]:
    """Get file info by media type"""
    try:
        from translation_orchestrator_agent.adk_agent_executor import get_file_info_from_context
        return get_file_info_from_context(media_type=media_type)
    except ImportError:
        return []

def get_files_by_extensions(extensions: List[str]) -> List[str]:
    """Get files by extensions from all available files"""
    all_files = get_file_paths_from_global_context()
    return [f for f in all_files if any(f.lower().endswith(ext) for ext in extensions)]

# TOOL FUNCTIONS - UPDATED AND COMPLETE

async def web_search_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Search the web for information"""
    result = await send_message_to_agent("web_search", query, original_user_query)
    return {"search_results": result["result"]} if result["status"] == "success" else {"error": result["error"]}

async def code_execution_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Execute code tasks and programming requests"""
    result = await send_message_to_agent("code", query, original_user_query)
    logger.info(f"Code execution result: {result}")
    
    if result["status"] == "success":
        return {
            "status": "completed",
            "code_result": result["result"],
            "response": result["result"],
            "final_answer": True  # This tells the orchestrator to stop and return
        }
    else:
        return {
            "status": "error",
            "error": result["error"],
            "response": f"Code execution failed: {result['error']}",
            "final_answer": True
        }

async def audio_conversational_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Process audio files and handle audio-related tasks"""
    # Get audio files
    audio_files = get_file_info_by_media_type("audio")
    file_paths = [f['path'] for f in audio_files if f.get('path')]
    
    # Extract audio paths from query if needed
    if not file_paths:
        import re
        pattern = r'["\']?([^"\']*\.(?:mp3|wav|flac|m4a|ogg|aac|wma))["\']?'
        matches = re.findall(pattern, query + " " + (original_user_query or ""), re.IGNORECASE)
        file_paths = [m for m in matches if os.path.exists(m)]
    
    # Enhanced message for audio processing
    message = f"Analyze audio and answer: {original_user_query or query}"
    if file_paths:
        message += f"\nAudio file: {file_paths[0]}"
    
    result = await send_message_to_agent("audio", message, original_user_query, file_paths=file_paths)
    
    if result["status"] == "success":
        return {
            "status": "completed",
            "audio_results": result["result"],
            "response": result["result"],
            "final_answer": True
        }
    else:
        return {
            "status": "error",
            "error": result["error"],
            "response": f"Audio processing failed: {result['error']}",
            "final_answer": True
        }

async def image_modification_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Modify and edit images (NOT analyze - use image_analyzer_function for analysis)"""
    image_files = get_file_info_by_media_type("image")
    file_paths = [f['path'] for f in image_files if f.get('path')]
    
    # Also try to get all files and filter for images
    if not file_paths:
        all_files = get_file_paths_from_global_context()
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg']
        file_paths = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    # Extract image paths from query
    if not file_paths:
        import re
        combined_query = f"{query} {original_user_query or ''}"
        pattern = r'["\']?([^"\']*\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|svg))["\']?'
        matches = re.findall(pattern, combined_query, re.IGNORECASE)
        file_paths = [m.strip() for m in matches if os.path.exists(m.strip())]
    
    # Enhanced message for image modification
    message = f"Modify/edit images as requested: {original_user_query or query}"
    if file_paths:
        message += f"\n\n[IMAGE_FILES]\n" + "\n".join([f"Image path: {path}" for path in file_paths])
    
    # FIXED: Call the correct 'image' agent (port 10014) instead of 'image_analyzer'
    result = await send_message_to_agent("image", message, original_user_query, file_paths=file_paths)
    
    if result["status"] == "success":
        return {
            "status": "completed",
            "image_modification_results": result["result"],
            "response": result["result"],
            "processed_files": file_paths,
            "final_answer": True
        }
    else:
        return {
            "status": "error",
            "error": result["error"],
            "response": f"Image modification failed: {result['error']}",
            "final_answer": True
        }

async def excel_file_analysis_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Analyze Excel files and spreadsheets using Claude's Code Execution tool"""
    excel_files = get_file_info_by_media_type("excel")
    file_paths = [f['path'] for f in excel_files if f.get('path')]
    
    # Also try to get all files and filter for Excel
    if not file_paths:
        all_files = get_file_paths_from_global_context()
        excel_extensions = ['.xlsx', '.xls', '.csv', '.tsv']
        file_paths = [f for f in all_files if any(f.lower().endswith(ext) for ext in excel_extensions)]
    
    # Extract Excel paths from query
    if not file_paths:
        import re
        combined_query = f"{query} {original_user_query or ''}"
        pattern = r'["\']?([^"\']*\.(?:xlsx|xls|csv|tsv))["\']?'
        matches = re.findall(pattern, combined_query, re.IGNORECASE)
        file_paths = [m.strip() for m in matches if os.path.exists(m.strip())]
    
    # Check if this is a visualization request
    visualization_keywords = ['chart', 'graph', 'plot', 'visualization', 'diagram', 'visualize', 'show chart', 'create chart']
    is_visualization_request = any(keyword in query.lower() or keyword in (original_user_query or "").lower() 
                                 for keyword in visualization_keywords)
    
    # Enhanced message for Excel analysis - match the agent's tool expectations
    if is_visualization_request:
        # For visualization requests, format message for create_visualization_only tool
        if file_paths:
            message = f"Please create a visualization using the file at: {file_paths[0]}. {original_user_query or query}"
        else:
            message = f"Please create a visualization: {original_user_query or query}"
    else:
        # For data analysis requests, format message for analyze_spreadsheet_with_claude tool
        if file_paths:
            message = f"Please analyze the spreadsheet file at: {file_paths[0]}. User request: {original_user_query or query}"
        else:
            message = f"Please analyze spreadsheet data: {original_user_query or query}"
    
    result = await send_message_to_agent("excel", message, original_user_query, file_paths=file_paths)
    
    if result["status"] == "success":
        return {
            "status": "completed",
            "excel_results": result["result"],
            "response": result["result"],
            "processed_files": file_paths,
            "final_answer": True
        }
    else:
        return {
            "status": "error",
            "error": result["error"],
            "response": f"Excel analysis failed: {result['error']}",
            "final_answer": True
        }

async def report_and_content_generation_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Generate reports and content"""
    all_files = get_file_paths_from_global_context()
    result = await send_message_to_agent("report", query, original_user_query, file_paths=all_files)
    
    if result["status"] == "success":
        return {
            "status": "completed",
            "report_results": result["result"],
            "response": result["result"],
            "final_answer": True
        }
    else:
        return {
            "status": "error",
            "error": result["error"],
            "response": f"Report generation failed: {result['error']}",
            "final_answer": True
        }

async def rag_agent_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Run RAG agent for document retrieval and question answering"""
    rag_files = get_file_info_by_media_type("rag")
    file_paths = [f['path'] for f in rag_files if f.get('path')]
    
    # Also try to get all files and filter for RAG-compatible formats
    if not file_paths:
        all_files = get_file_paths_from_global_context()
        rag_extensions = ['.pdf', '.txt', '.docx', '.md', '.doc', '.rtf']
        file_paths = [f for f in all_files if any(f.lower().endswith(ext) for ext in rag_extensions)]
    
    # Extract document paths from query
    if not file_paths:
        import re
        combined_query = f"{query} {original_user_query or ''}"
        pattern = r'["\']?([^"\']*\.(?:pdf|txt|docx|md|doc|rtf))["\']?'
        matches = re.findall(pattern, combined_query, re.IGNORECASE)
        file_paths = [m.strip() for m in matches if os.path.exists(m.strip())]
    
    # Enhanced message for RAG processing
    message = f"Process documents and answer: {original_user_query or query}"
    if file_paths:
        message += f"\n\n[DOCUMENTS]\n" + "\n".join([f"Document path: {path}" for path in file_paths])
    
    result = await send_message_to_agent("rag", message, original_user_query, file_paths=file_paths)
    
    if result["status"] == "success":
        return {
            "status": "completed",
            "rag_results": result["result"],
            "response": result["result"],
            "processed_files": file_paths,
            "final_answer": True
        }
    else:
        return {
            "status": "error",
            "error": result["error"],
            "response": f"RAG processing failed: {result['error']}",
            "final_answer": True
        }

async def image_analyzer_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Analyze images and provide detailed insights"""
    image_files = get_file_info_by_media_type("image")
    file_paths = [f['path'] for f in image_files if f.get('path')]
    
    # Also try to get all files and filter for images
    if not file_paths:
        all_files = get_file_paths_from_global_context()
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg']
        file_paths = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    # Extract image paths from query
    if not file_paths:
        import re
        combined_query = f"{query} {original_user_query or ''}"
        pattern = r'["\']?([^"\']*\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|svg))["\']?'
        matches = re.findall(pattern, combined_query, re.IGNORECASE)
        file_paths = [m.strip() for m in matches if os.path.exists(m.strip())]
    
    # Enhanced message for image analysis
    message = f"Analyze images and answer: {original_user_query or query}"
    if file_paths:
        message += f"\n\n[IMAGE_FILES]\n" + "\n".join([f"Image path: {path}" for path in file_paths])
    
    result = await send_message_to_agent("image_analyzer", message, original_user_query, file_paths=file_paths)
    
    if result["status"] == "success":
        return {
            "status": "completed",
            "image_analysis_results": result["result"],
            "response": result["result"],
            "processed_files": file_paths,
            "final_answer": True
        }
    else:
        return {
            "status": "error",
            "error": result["error"],
            "response": f"Image analysis failed: {result['error']}",
            "final_answer": True
        }

async def image_generation_function(query: str, original_user_query: str) -> Dict[str, Any]:
    result = await send_message_to_agent("image_generation",query, original_user_query)
    
    if result["status"] == "success":
        return {
            "status": "completed",
            "image_generation_results": result["result"],
            "response": result["result"],
            "final_answer": True
        }
    else:
        return {
            "status": "error",
            "error": result["error"],
            "response": f"Image generation failed: {result['error']}",
            "final_answer": True
        }

async def video_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Process video files and handle video-related tasks"""
    video_files = get_file_info_by_media_type("video")
    file_paths = [f['path'] for f in video_files if f.get('path')]
    
    # Also try to get all files and filter for videos
    if not file_paths:
        all_files = get_file_paths_from_global_context()
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
        file_paths = [f for f in all_files if any(f.lower().endswith(ext) for ext in video_extensions)]
    
    # Extract video paths from query
    if not file_paths:
        import re
        combined_query = f"{query} {original_user_query or ''}"
        pattern = r'["\']?([^"\']*\.(?:mp4|avi|mov|mkv|webm|flv|wmv|m4v))["\']?'
        matches = re.findall(pattern, combined_query, re.IGNORECASE)
        file_paths = [m.strip() for m in matches if os.path.exists(m.strip())]
    
    # Enhanced message for video processing
    message = f"Process video and answer: {original_user_query or query}"
    if file_paths:
        message += f"\n\n[VIDEO_FILES]\n" + "\n".join([f"Video path: {path}" for path in file_paths])
    
    result = await send_message_to_agent("video", message, original_user_query, file_paths=file_paths)
    
    if result["status"] == "success":
        return {
            "status": "completed",
            "video_results": result["result"],
            "response": result["result"],
            "processed_files": file_paths,
            "final_answer": True
        }
    else:
        return {
            "status": "error",
            "error": result["error"],
            "response": f"Video processing failed: {result['error']}",
            "final_answer": True
        }

# Backwards compatibility (deprecated)
def inject_multimedia_context_for_agent(agent_type: str, multimedia_parts: Optional[List] = None):
    """DEPRECATED: Maintained for backwards compatibility"""
    pass

_current_multimedia_context = None

async def send_message_to_agent_with_context(agent_type: str, message: str, original_user_query: str, timeout: float = REQUEST_TIMEOUT) -> Dict[str, Any]:
    """DEPRECATED: Use send_message_to_agent directly"""
    return await send_message_to_agent(agent_type, message, original_user_query)