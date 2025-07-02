# a2a_translation_tools.py - FIXED VERSION WITH JSON SERIALIZATION
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

# Agent Configuration
AGENTS = {
    "web_search": {"url": "http://localhost:10012", "id": "web_searcher"},
    "code": {"url": "http://localhost:10011", "id": "code_executor"},
    "audio": {"url": "http://localhost:10013", "id": "audio_processor"},
    "image": {"url": "http://localhost:10014", "id": "img_to_img_processor"},
    "video": {"url": "http://localhost:10016", "id": "video_processor"},
    "report": {"url": "http://localhost:10015", "id": "report_content_generator"}
}

REQUEST_TIMEOUT = 120.0
POLL_INTERVAL = 5.0
MAX_POLL_ATTEMPTS = 50

# JSON SERIALIZATION FIX
class EnumJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Enum objects"""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

def serialize_enum_objects(obj):
    """Recursively convert Enum objects to their values for JSON serialization"""
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {key: serialize_enum_objects(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_enum_objects(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle objects with attributes
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                result[key] = serialize_enum_objects(value)
        return result
    elif hasattr(obj, 'model_dump'):
        # Handle Pydantic models
        return serialize_enum_objects(obj.model_dump())
    elif hasattr(obj, 'dict'):
        # Handle other dict-like objects
        return serialize_enum_objects(obj.dict())
    else:
        return obj

def safe_json_dumps(obj, **kwargs):
    """Safely serialize objects to JSON, handling Enums"""
    try:
        # First try with custom encoder
        return json.dumps(obj, cls=EnumJSONEncoder, **kwargs)
    except (TypeError, ValueError):
        # If that fails, serialize enum objects first
        serialized_obj = serialize_enum_objects(obj)
        return json.dumps(serialized_obj, **kwargs)

async def check_agent_availability(agent_url: str) -> bool:
    """Check if an agent is available"""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            response = await client.get(f"{agent_url}/health")
            return response.status_code == 200
    except:
        return False
    


async def poll_task_completion(client: A2AClient, task_id: str) -> Optional[Task]:
    """Poll task until completion"""
    for _ in range(MAX_POLL_ATTEMPTS):
        try:
            query_params = TaskQueryParams(id=task_id)
            response = await client.get_task(GetTaskRequest(params=query_params))
            
            if isinstance(response.root, GetTaskSuccessResponse):
                task = response.root.result
                if task.status.state in [TaskState.completed, TaskState.failed, TaskState.canceled]:
                    return task
            
            await asyncio.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.warning(f"Polling error: {e}")
    
    return None

async def send_message_to_agent(
    agent_type: str, 
    message: str, 
    original_user_query: str,
    file_paths: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Send message to agent and return response - FIXED VERSION"""
    
    if agent_type not in AGENTS:
        return {"status": "error", "error": f"Unknown agent: {agent_type}"}
    
    agent_config = AGENTS[agent_type]
    if not await check_agent_availability(agent_config["url"]):
        return {"status": "error", "error": f"Agent {agent_type} unavailable"}
    
    try:
        # Enhance message with file info
        enhanced_message = message
        if file_paths:
            enhanced_message += f"\n\n[FILES]\n" + "\n".join([f"Path: {p}" for p in file_paths])
        if file_urls:
            enhanced_message += f"\n\n[URLS]\n" + "\n".join([f"URL: {u}" for u in file_urls])
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as httpx_client:
            client = A2AClient(url=agent_config["url"], httpx_client=httpx_client)
            
            # CRITICAL FIX: Use string value for role instead of enum
            payload = {
                'message': {
                    'role': 'user',  # Use string directly instead of enum
                    'parts': [{'kind': 'text', 'text': enhanced_message}],
                    'messageId': uuid4().hex,
                },
                'agentId': agent_config["id"],
                'userId': "orchestrator",
            }
            
            # Additional safety: serialize enum objects in payload
            serialized_payload = serialize_enum_objects(payload)
            
            request = SendMessageRequest(params=MessageSendParams(**serialized_payload))
            
            
            # Log the request for debugging
            logger.info(f"Sending request to {agent_type}: {safe_json_dumps(serialized_payload, indent=2)[:500]}...")
            
            response = await client.send_message(request)
            logger.info(f"Response from {agent_type} agent: {response}")
            
            if isinstance(response.root, SendMessageSuccessResponse):
                result = response.root.result
                logger.info(f"Received result from {agent_type}: {result}")
                
                if isinstance(result, Task):
                    # Poll for completion
                    completed_task = await poll_task_completion(client, result.id)
                    logger.info(f"Task {completed_task}")
                    if completed_task and completed_task.status.state == TaskState.completed:
                        # FIX: Extract text properly from completed task
                        response_text = "Task completed successfully"
                        if hasattr(completed_task, 'artifacts') and completed_task.artifacts:
                            # Extract text from artifacts if available
                            for artifact in completed_task.artifacts:
                                if hasattr(artifact, 'parts'):
                                    for part in artifact.parts:
                                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                            response_text = part.root.text
                                            break
                                        elif hasattr(part, 'text'):
                                            response_text = part.text
                                            break
                                    if response_text != "Task completed successfully":
                                        break
                        return {"status": "success", "result": response_text}
                    else:
                        return {"status": "error", "error": "Task failed or timed out"}
                
                elif isinstance(result, Message):
                    # Direct response - FIX: Extract text properly from Message
                    response_text = "Response received"
                    if hasattr(result, 'parts') and result.parts:
                        for part in result.parts:
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                response_text = part.root.text
                                break
                            elif hasattr(part, 'text'):
                                response_text = part.text
                                break
                    elif hasattr(result, 'text'):
                        response_text = result.text
                    
                    return {"status": "success", "result": response_text}
            
            return {"status": "error", "error": "Unexpected response format"}
            
    except Exception as e:
        logger.error(f"Error sending message to {agent_type}: {e}")
        return {"status": "error", "error": str(e)}

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

# TOOL FUNCTIONS

async def web_search_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Search the web"""
    result = await send_message_to_agent("web_search", query, original_user_query)
    return {"search_results": result["result"]} if result["status"] == "success" else {"error": result["error"]}

async def code_execution_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Execute code tasks"""
    result = await send_message_to_agent("code", query, original_user_query)
    logger.info(f"Code execution result: {result}")
    return {"code_result": result["result"]} if result["status"] == "success" else {"error": result["error"]}

async def audio_conversational_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Process audio files"""
    # Get audio files
    audio_files = get_file_info_by_media_type("audio")
    file_paths = [f['path'] for f in audio_files if f.get('path')]
    
    # Extract audio paths from query if needed
    if not file_paths:
        import re
        pattern = r'["\']?([^"\']*\.(?:mp3|wav|flac|m4a|ogg|aac))["\']?'
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
    """Process images - FIXED VERSION"""
    
    # DEBUG: Check what's in the global context
    print(f"DEBUG: Checking global multimedia context...")
    
    # Get image files from context - PRIORITY 1
    image_files = get_file_info_by_media_type("image")
    file_paths = [f['path'] for f in image_files if f.get('path')]
    print(f"DEBUG: Image files from context: {image_files}")
    print(f"DEBUG: Extracted paths: {file_paths}")
    
    # Also try to get all files and filter for images - PRIORITY 2
    if not file_paths:
        all_files = get_file_paths_from_global_context()
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']
        file_paths = [f for f in all_files if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    # Extract file paths from query text - PRIORITY 3 (only if no paths found yet)
    file_urls = []
    clean_query = query
    
    if not file_paths:
        import re
        
        # Use sets to avoid duplicates
        found_paths = set()
        found_urls = set()
        
        # Combine all text to search
        search_text = f"{query} {original_user_query or ''}"
        
        # More targeted regex patterns - prioritize specificity
        path_patterns = [
            # Windows absolute paths (highest priority)
            r'["\']?([a-zA-Z]:[\\\/](?:[^\\\/\s:*?"<>|]+[\\\/])*[^\\\/\s:*?"<>|]+\.(?:jpg|jpeg|png|gif|bmp|webp|tiff))["\']?',
            # Unix absolute paths
            r'["\']?(\/(?:[^\/\s]+\/)*[^\/\s]+\.(?:jpg|jpeg|png|gif|bmp|webp|tiff))["\']?',
            # HTTP URLs
            r'(https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|gif|bmp|webp|tiff))',
            # Simple filenames (lowest priority)
            r'["\']?([^"\'\s\/\\:*?"<>|]+\.(?:jpg|jpeg|png|gif|bmp|webp|tiff))["\']?'
        ]
        
        # Process patterns in order of priority
        for i, pattern in enumerate(path_patterns):
            matches = re.findall(pattern, search_text, re.IGNORECASE)
            for match in matches:
                # Clean up the match
                match = match.strip().strip('"\'')
                
                if match.startswith(('http://', 'https://')):
                    found_urls.add(match)
                elif os.path.exists(match):
                    found_paths.add(match)
                    # Remove this path from clean_query to avoid processing it as text
                    clean_query = clean_query.replace(match, '').strip()
                elif i == 0:  # For Windows paths, try both forward and back slashes
                    # Try converting slashes
                    alt_match = match.replace('/', '\\') if '/' in match else match.replace('\\', '/')
                    if os.path.exists(alt_match):
                        found_paths.add(alt_match)
                        clean_query = clean_query.replace(match, '').strip()
        
        # Convert sets back to lists
        file_paths = list(found_paths)
        file_urls = list(found_urls)
    
    # Remove duplicates from file_paths (in case context had duplicates)
    file_paths = list(dict.fromkeys(file_paths))  # Preserves order while removing duplicates
    file_urls = list(dict.fromkeys(file_urls))
    
    # Debug logging
    print(f"DEBUG: Found {len(file_paths)} unique image files")
    print(f"DEBUG: Found {len(file_urls)} unique image URLs")
    print(f"DEBUG: Final image files: {file_paths}")
    print(f"DEBUG: Final image URLs: {file_urls}")
    
    # Use original query if clean query is too short
    if len(clean_query.strip()) < 3:
        clean_query = original_user_query or "Process this image"
    
    # Enhanced message with explicit file information
    message = clean_query
    if file_paths:
        message += f"\n\n[IMAGE_FILES]\n" + "\n".join([f"Image path: {path}" for path in file_paths])
    if file_urls:
        message += f"\n\n[IMAGE_URLS]\n" + "\n".join([f"Image URL: {url}" for url in file_urls])
    
    # If no files found, return error immediately
    if not file_paths and not file_urls:
        return {
            "status": "error", 
            "error": "No image files found. Please ensure the image file is properly uploaded or provide a valid image path.",
            "debug_info": {
                "original_query": original_user_query,
                "query": query,
                "context_files": get_file_paths_from_global_context(),
                "image_context": get_file_info_by_media_type("image")
            }
        }
    
    print(f"DEBUG: Sending message to agent: {message[:200]}...")
    
    result = await send_message_to_agent("image", message, original_user_query)
    
    if result["status"] == "success":
        return {
            "status": "completed",
            "image_results": result["result"],
            "response": result["result"],
            "processed_files": file_paths + file_urls,
            "final_answer": True
        }
    else:
        return {
            "status": "error",
            "error": result["error"],
            "response": f"Image processing failed: {result['error']}",
            "final_answer": True
        }
        
        
async def report_and_content_generation_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Generate reports and content"""
    all_files = get_file_paths_from_global_context()
    result = await send_message_to_agent("report", query, original_user_query, file_paths=all_files)
    return {"report_results": result["result"]} if result["status"] == "success" else {"error": result["error"]}

# Backwards compatibility (deprecated)
def inject_multimedia_context_for_agent(agent_type: str, multimedia_parts: Optional[List] = None):
    """DEPRECATED: Maintained for backwards compatibility"""
    pass

_current_multimedia_context = None

async def send_message_to_agent_with_context(agent_type: str, message: str, original_user_query: str, timeout: float = REQUEST_TIMEOUT) -> Dict[str, Any]:
    """DEPRECATED: Use send_message_to_agent directly"""
    return await send_message_to_agent(agent_type, message, original_user_query)

