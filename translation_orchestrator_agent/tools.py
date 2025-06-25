# a2a_translation_tools.py - ADK-COMPATIBLE VERSION WITH DIRECT PATH SUPPORT
import asyncio
import httpx
import logging
from typing import Dict, Any, Optional, List
from uuid import uuid4
import time
from a2a.client import A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendMessageSuccessResponse,
    Task,
    Message,
    TextPart,
    TaskState,
    GetTaskRequest,
    GetTaskSuccessResponse,
    TaskQueryParams,
)

logger = logging.getLogger(__name__)

# Agent Configuration
AGENTS = {
    "web_search": {
        "url": "http://localhost:10012",
        "id": "web_searcher"
    },
    "code": {
        "url": "http://localhost:10011", 
        "id": "code_executor"
    },
    "audio": {
        "url": "http://localhost:10013",
        "id": "audio_processor"
    },
    "image": {
        "url": "http://localhost:10014",
        "id": "img_to_img_processor"
    },
    "video": {
        "url": "http://localhost:10016",
        "id": "video_processor"
    },
    "report": {
        "url": "http://localhost:10015",
        "id": "report_content_generator"
    }
}

REQUEST_TIMEOUT = 120.0
POLL_INTERVAL = 1.0
MAX_POLL_ATTEMPTS = 120

async def check_agent_availability(agent_url: str, agent_id: str) -> bool:
    """Check if an agent is available"""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            response = await client.get(f"{agent_url}/health")
            return response.status_code == 200
    except Exception as e:
        logger.warning(f"Agent {agent_id} unavailable: {e}")
        return False

def debug_task_structure(task_or_message, name="object"):
    """Debug function to understand the exact structure of Task objects"""
    logger.info(f"=== DEBUGGING {name.upper()} STRUCTURE ===")
    logger.info(f"Type: {type(task_or_message)}")
    
    # Log all attributes
    attributes = [attr for attr in dir(task_or_message) if not attr.startswith('_')]
    logger.info(f"Available attributes: {attributes}")
    
    # Check each attribute
    for attr in attributes:
        try:
            value = getattr(task_or_message, attr)
            if value is not None:
                logger.info(f"{attr}: {type(value)} - {str(value)[:100]}...")
                
                # If it's a list, check its contents
                if isinstance(value, list) and value:
                    logger.info(f"  {attr} list contents:")
                    for i, item in enumerate(value[:3]):  # Only first 3 items
                        logger.info(f"    [{i}]: {type(item)} - {str(item)[:100]}...")
                        
                        # If item has parts, check those too
                        if hasattr(item, 'parts') and item.parts:
                            logger.info(f"      {attr}[{i}].parts:")
                            for j, part in enumerate(item.parts[:2]):  # Only first 2 parts
                                logger.info(f"        [{j}]: {type(part)} - {str(part)[:100]}...")
                                if hasattr(part, 'root'):
                                    logger.info(f"          root: {type(part.root)} - {str(part.root)[:100]}...")
        except Exception as e:
            logger.warning(f"Error accessing {attr}: {e}")
    
    logger.info(f"=== END DEBUGGING {name.upper()} ===")
    
async def poll_task_until_completion(client: A2AClient, task_id: str, max_attempts: int = MAX_POLL_ATTEMPTS) -> Optional[Task]:
    """Polls task until completion with enhanced debugging"""
    attempts = 0
    
    while attempts < max_attempts:
        try:
            query_params = TaskQueryParams(id=task_id)
            task_response = await client.get_task(GetTaskRequest(params=query_params))
            
            if isinstance(task_response.root, GetTaskSuccessResponse) and isinstance(task_response.root.result, Task):
                task = task_response.root.result
                
                # DEBUG: Log task structure on completion
                if task.status.state in [TaskState.completed, TaskState.failed, TaskState.canceled]:
                    debug_task_structure(task, f"completed_task_{task_id}")
                    return task
                    
            attempts += 1
            await asyncio.sleep(POLL_INTERVAL)
            
        except Exception as e:
            logger.warning(f"Error polling task {task_id}: {e}")
            attempts += 1
    
    return None

def aggressive_text_extraction(obj) -> str:
    """Aggressive text extraction as last resort"""
    try:
        import json
        
        # Convert object to dict if possible
        if hasattr(obj, '__dict__'):
            obj_dict = obj.__dict__
        elif hasattr(obj, 'model_dump'):
            obj_dict = obj.model_dump()
        elif hasattr(obj, 'dict'):
            obj_dict = obj.dict()
        else:
            obj_dict = {}
        
        # Recursively search for text content
        def find_text_in_dict(d, path=""):
            texts = []
            if isinstance(d, dict):
                for key, value in d.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(value, str) and len(value.strip()) > 10:  # Only meaningful text
                        if not any(skip in key.lower() for skip in ['id', 'url', 'type', 'role', 'timestamp']):
                            texts.append(value.strip())
                            logger.info(f"Found text at {current_path}: {len(value)} characters")
                    elif isinstance(value, (dict, list)):
                        texts.extend(find_text_in_dict(value, current_path))
            elif isinstance(d, list):
                for i, item in enumerate(d):
                    current_path = f"{path}[{i}]" if path else f"[{i}]"
                    texts.extend(find_text_in_dict(item, current_path))
            return texts
        
        found_texts = find_text_in_dict(obj_dict)
        result = "\n\n".join(found_texts)
        
        if result:
            logger.info(f"Aggressive extraction found {len(result)} characters")
        else:
            logger.warning("Aggressive extraction found no text")
            # Log the structure for debugging
            try:
                structure = json.dumps(obj_dict, default=str, indent=2)[:1000]
                logger.info(f"Object structure (first 1000 chars): {structure}")
            except:
                logger.info(f"Object type: {type(obj)}, attributes: {dir(obj)[:10]}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in aggressive text extraction: {e}")
        return ""

def extract_all_text_content(task_or_message) -> str:
    """Enhanced text extraction with better function response handling - FIXED VERSION"""
    result_texts = []
    
    try:
        # Handle Task objects
        if isinstance(task_or_message, Task):
            logger.info(f"Processing Task object with status: {task_or_message.status.state if hasattr(task_or_message, 'status') else 'unknown'}")
            
            # PRIORITY 1: Check for function responses in messages first
            if hasattr(task_or_message, 'messages') and task_or_message.messages:
                for message in task_or_message.messages:
                    if hasattr(message, 'role') and message.role in ['assistant', 'model']:
                        # Look for function responses which contain the actual results
                        if hasattr(message, 'parts') and message.parts:
                            for part in message.parts:
                                if hasattr(part, 'function_response') and part.function_response:
                                    func_resp = part.function_response
                                    if hasattr(func_resp, 'response') and func_resp.response:
                                        # Extract the actual response content
                                        if isinstance(func_resp.response, dict):
                                            # Look for various response keys
                                            for key in ['audio_results', 'result', 'response', 'content', 'answer']:
                                                if key in func_resp.response and func_resp.response[key]:
                                                    content = func_resp.response[key]
                                                    if isinstance(content, str):
                                                        result_texts.append(content.strip())
                                                        logger.info(f"Extracted from function_response.{key}: {len(content)} chars")
                                                    break
                                        elif isinstance(func_resp.response, str):
                                            result_texts.append(func_resp.response.strip())
                                            logger.info(f"Extracted from function_response (string): {len(func_resp.response)} chars")
            
            # PRIORITY 2: Method 1: Extract from artifacts (if no function response found)
            if not result_texts and hasattr(task_or_message, 'artifacts') and task_or_message.artifacts:
                logger.info(f"Found {len(task_or_message.artifacts)} artifacts")
                for i, artifact in enumerate(task_or_message.artifacts):
                    logger.info(f"Processing artifact {i}: {type(artifact)}")
                    
                    # Check if artifact has content directly
                    if hasattr(artifact, 'content') and artifact.content:
                        if isinstance(artifact.content, str):
                            result_texts.append(artifact.content.strip())
                            logger.info(f"Extracted text from artifact.content: {len(artifact.content)} characters")
                    
                    # Check artifact parts
                    if hasattr(artifact, 'parts') and artifact.parts:
                        for j, part in enumerate(artifact.parts):
                            logger.info(f"Processing artifact {i}, part {j}: {type(part)}")
                            
                            # Direct text access
                            if hasattr(part, 'text') and part.text:
                                result_texts.append(part.text.strip())
                                logger.info(f"Extracted text from part.text: {len(part.text)} characters")
                            
                            # Root-based text access
                            elif hasattr(part, 'root'):
                                if isinstance(part.root, TextPart) and part.root.text:
                                    result_texts.append(part.root.text.strip())
                                    logger.info(f"Extracted text from part.root.text: {len(part.root.text)} characters")
                                elif hasattr(part.root, 'text') and part.root.text:
                                    result_texts.append(part.root.text.strip())
                                    logger.info(f"Extracted text from part.root.text: {len(part.root.text)} characters")
                                elif isinstance(part.root, str):
                                    result_texts.append(part.root.strip())
                                    logger.info(f"Extracted text from part.root (string): {len(part.root)} characters")
            
            # Continue with other extraction methods if still no content...
            # [Rest of the original extraction logic remains the same]
            
        # Handle Message objects with function responses
        elif isinstance(task_or_message, Message):
            logger.info("Processing Message object")
            
            # Check for function responses first
            if hasattr(task_or_message, 'parts') and task_or_message.parts:
                for part in task_or_message.parts:
                    if hasattr(part, 'function_response') and part.function_response:
                        func_resp = part.function_response
                        if hasattr(func_resp, 'response') and func_resp.response:
                            if isinstance(func_resp.response, dict):
                                for key in ['audio_results', 'result', 'response', 'content']:
                                    if key in func_resp.response and func_resp.response[key]:
                                        content = func_resp.response[key]
                                        if isinstance(content, str):
                                            result_texts.append(content.strip())
                                            logger.info(f"Extracted from message function_response.{key}: {len(content)} chars")
                                        break
            
            # [Continue with original Message handling logic if no function response...]
        
        # Join all extracted text
        full_text = "\n\n".join(result_texts)
        logger.info(f"Total extracted text: {len(full_text)} characters")
        
        return full_text
        
    except Exception as e:
        logger.error(f"Error in extract_all_text_content: {e}")
        return ""

# FIXED: Agent communication with better response handling
async def send_message_to_agent(
    agent_type: str,
    message: str,
    original_user_query: str,
    file_paths: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None,
    timeout: float = REQUEST_TIMEOUT
) -> Dict[str, Any]:
    """Send message to any agent with enhanced response handling - FIXED VERSION"""
    
    if agent_type not in AGENTS:
        return {"status": "error", "error_message": f"Unknown agent type: {agent_type}"}
    
    agent_config = AGENTS[agent_type]
    agent_url = agent_config["url"]
    agent_id = agent_config["id"]
    
    if not await check_agent_availability(agent_url, agent_id):
        return {
            "status": "error", 
            "error_message": f"Agent {agent_id} is not available"
        }
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as httpx_client:
            client = A2AClient(url=agent_url, httpx_client=httpx_client)
            
            # Create enhanced message with file path/URL information
            enhanced_message = message
            
            # Add file path information to the message
            if file_paths:
                file_info = []
                for file_path in file_paths:
                    file_info.append(f"File path: {file_path}")
                
                enhanced_message = f"""{message}

[FILE PATHS AVAILABLE]
{chr(10).join(file_info)}

Note: These are direct file paths that can be read by the agent."""
            
            # Add file URL information to the message
            if file_urls:
                url_info = []
                for file_url in file_urls:
                    url_info.append(f"File URL: {file_url}")
                
                enhanced_message = f"""{enhanced_message}

[FILE URLS AVAILABLE]
{chr(10).join(url_info)}

Note: These are URLs that can be downloaded by the agent."""
            
            # Simple text-only message creation
            message_parts = [{'kind': 'text', 'text': enhanced_message}]
            
            send_message_payload: dict[str, Any] = {
                'message': {
                    'role': 'user',
                    'parts': message_parts,
                    'messageId': uuid4().hex,
                },
                'agentId': agent_id,
                'userId': "orchestrator",
            }
            
            request = SendMessageRequest(params=MessageSendParams(**send_message_payload))
            start_time = time.time()
            response = await client.send_message(request)
            
            if isinstance(response.root, SendMessageSuccessResponse):
                if isinstance(response.root.result, Task):
                    task_id = response.root.result.id
                    logger.info(f"Polling task {task_id} for completion...")
                    completed_task = await poll_task_until_completion(client, task_id)
                    
                    if completed_task and completed_task.status.state == TaskState.completed:
                        # Use enhanced text extraction
                        result_text = extract_all_text_content(completed_task)
                        
                        execution_time = time.time() - start_time
                        
                        if not result_text.strip():
                            logger.warning("No text content extracted from completed task - checking task structure")
                            debug_task_structure(completed_task, f"empty_result_task_{task_id}")
                            result_text = "Task completed but no readable content was extracted"
                        
                        return {
                            "status": "success",
                            "result": result_text,
                            "execution_time": execution_time,
                            "agent_used": agent_id
                        }
                    else:
                        error_msg = "Task failed or timed out"
                        if completed_task:
                            error_msg += f" (final state: {completed_task.status.state})"
                        return {"status": "error", "error_message": error_msg}
                
                elif isinstance(response.root.result, Message):
                    # Use enhanced text extraction for direct message responses
                    result_text = extract_all_text_content(response.root.result)
                    
                    execution_time = time.time() - start_time
                    
                    if not result_text.strip():
                        logger.warning("No text content extracted from message response")
                        debug_task_structure(response.root.result, "empty_message_result")
                        result_text = "Response received but no readable content was extracted"
                    
                    return {
                        "status": "success", 
                        "result": result_text,
                        "execution_time": execution_time,
                        "agent_used": agent_id
                    }
            
            return {"status": "error", "error_message": "Unexpected response format"}
            
    except Exception as e:
        logger.error(f"Error communicating with {agent_id}: {e}")
        return {"status": "error", "error_message": f"Communication error: {str(e)}"}
    
    
async def send_message_to_agent(
    agent_type: str,
    message: str,
    original_user_query: str,
    file_paths: Optional[List[str]] = None,
    file_urls: Optional[List[str]] = None,
    timeout: float = REQUEST_TIMEOUT
) -> Dict[str, Any]:
    """Send message to any agent with direct file path/URL support"""
    
    if agent_type not in AGENTS:
        return {"status": "error", "error_message": f"Unknown agent type: {agent_type}"}
    
    agent_config = AGENTS[agent_type]
    agent_url = agent_config["url"]
    agent_id = agent_config["id"]
    
    if not await check_agent_availability(agent_url, agent_id):
        return {
            "status": "error", 
            "error_message": f"Agent {agent_id} is not available"
        }
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as httpx_client:
            client = A2AClient(url=agent_url, httpx_client=httpx_client)
            
            # Create enhanced message with file path/URL information
            enhanced_message = message
            
            # Add file path information to the message
            if file_paths:
                file_info = []
                for file_path in file_paths:
                    file_info.append(f"File path: {file_path}")
                
                enhanced_message = f"""{message}

[FILE PATHS AVAILABLE]
{chr(10).join(file_info)}

Note: These are direct file paths that can be read by the agent."""
            
            # Add file URL information to the message
            if file_urls:
                url_info = []
                for file_url in file_urls:
                    url_info.append(f"File URL: {file_url}")
                
                enhanced_message = f"""{enhanced_message}

[FILE URLS AVAILABLE]
{chr(10).join(url_info)}

Note: These are URLs that can be downloaded by the agent."""
            
            # Simple text-only message creation (no binary data)
            message_parts = [{'kind': 'text', 'text': enhanced_message}]
            
            send_message_payload: dict[str, Any] = {
                'message': {
                    'role': 'user',
                    'parts': message_parts,
                    'messageId': uuid4().hex,
                },
                'agentId': agent_id,
                'userId': "orchestrator",
            }
            
            request = SendMessageRequest(params=MessageSendParams(**send_message_payload))
            start_time = time.time()
            response = await client.send_message(request)
            
            if isinstance(response.root, SendMessageSuccessResponse):
                if isinstance(response.root.result, Task):
                    task_id = response.root.result.id
                    logger.info(f"Polling task {task_id} for completion...")
                    completed_task = await poll_task_until_completion(client, task_id)
                    
                    if completed_task and completed_task.status.state == TaskState.completed:
                        # Use enhanced text extraction
                        result_text = extract_all_text_content(completed_task)
                        
                        execution_time = time.time() - start_time
                        
                        if not result_text.strip():
                            logger.warning("No text content extracted from completed task")
                            result_text = "Task completed but no content returned"
                        
                        return {
                            "status": "success",
                            "result": result_text,
                            "execution_time": execution_time,
                            "agent_used": agent_id
                        }
                    else:
                        error_msg = "Task failed or timed out"
                        if completed_task:
                            error_msg += f" (final state: {completed_task.status.state})"
                        return {"status": "error", "error_message": error_msg}
                
                elif isinstance(response.root.result, Message):
                    # Use enhanced text extraction for direct message responses
                    result_text = extract_all_text_content(response.root.result)
                    
                    execution_time = time.time() - start_time
                    
                    if not result_text.strip():
                        logger.warning("No text content extracted from message response")
                        result_text = "Response received but no content extracted"
                    
                    return {
                        "status": "success", 
                        "result": result_text,
                        "execution_time": execution_time,
                        "agent_used": agent_id
                    }
            
            return {"status": "error", "error_message": "Unexpected response format"}
            
    except Exception as e:
        logger.error(f"Error communicating with {agent_id}: {e}")
        return {"status": "error", "error_message": f"Communication error: {str(e)}"}

# Helper function to extract file paths from global multimedia context
def get_file_paths_from_global_context() -> List[str]:
    """Extract file paths from global multimedia context"""
    try:
        # Import the global context from the executor module
        from orchestrator_agent.adk_executor import get_file_paths_from_context
        return get_file_paths_from_context()
    except ImportError:
        logger.warning("Could not import global context - using fallback method")
        return []

def get_file_info_by_media_type(media_type: str) -> List[Dict]:
    """Get file information filtered by media type"""
    try:
        from orchestrator_agent.adk_executor import get_file_info_from_context
        return get_file_info_from_context(media_type=media_type)
    except ImportError:
        logger.warning("Could not import global context - using fallback method")
        return []

# ADK-COMPATIBLE TOOL FUNCTIONS - ENHANCED WITH DIRECT PATH SUPPORT

async def web_search_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Search the web for information"""
    logger.info(f"Web search: '{query}'")
    
    result = await send_message_to_agent("web_search", query, original_user_query)
    
    if result["status"] == "success":
        logger.info(f"Web search successful: {len(result['result'])} characters returned")
        return {"search_results": result["result"]}
    else:
        logger.error(f"Web search failed: {result['error_message']}")
        return {"error": f"Search failed: {result['error_message']}"}

async def code_execution_function(code_or_task: str, original_user_query: str) -> Dict[str, Any]:
    """Execute code or handle coding tasks"""
    logger.info(f"Code task: '{code_or_task}'")
    
    result = await send_message_to_agent("code", code_or_task, original_user_query)
    
    if result["status"] == "success":
        logger.info(f"Code execution successful: {len(result['result'])} characters returned")
        return {"code_result": result["result"]}
    else:
        logger.error(f"Code execution failed: {result['error_message']}")
        return {"error": f"Code execution failed: {result['error_message']}"}

async def audio_conversational_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Process audio files and audio-related tasks with direct path support - FIXED VERSION"""
    logger.info(f"Audio task: '{query}'")
    
    # Get audio file paths from global context
    try:
        audio_files = get_file_info_by_media_type("audio")
        file_paths = [f['path'] for f in audio_files if f.get('path')]
    except Exception as e:
        logger.warning(f"Could not get audio files from context: {e}")
        file_paths = []
        
        # Extract file path from query if present
        if query and ('\\' in query or '/' in query):
            # The query itself might be a file path
            if query.endswith(('.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac')):
                file_paths = [query]
                logger.info(f"Extracted audio file path from query: {query}")
    
    logger.info(f"Found {len(file_paths)} audio files: {file_paths}")
    
    result = await send_message_to_agent(
        agent_type="audio", 
        message=query, 
        original_user_query=original_user_query,
        file_paths=file_paths
    )
    
    if result["status"] == "success":
        logger.info(f"Audio processing successful: {len(result['result'])} characters returned")
        
        # FIXED: Return a properly formatted response that the orchestrator can understand
        audio_response = result["result"]
        
        # Clean up the response if it has markdown formatting
        if audio_response.startswith("```") and audio_response.endswith("```"):
            # Remove markdown code block formatting
            lines = audio_response.split('\n')
            if lines[0].strip() == "```" or lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            audio_response = '\n'.join(lines)
        
        # Return the response in a format that clearly indicates completion
        return {
            "status": "completed",
            "audio_results": audio_response,
            "summary": "Audio processing completed successfully. The audio file has been analyzed and transcribed.",
            "final_answer": True  # Flag to indicate this is the final answer
        }
    else:
        logger.error(f"Audio processing failed: {result['error_message']}")
        return {
            "status": "error", 
            "error": f"Audio processing failed: {result['error_message']}",
            "final_answer": True  # Even errors should be final
        }

async def video_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Process video files and video-related tasks with direct path support"""
    logger.info(f"Video task: '{query}'")
    
    # Get video file paths from global context
    video_files = get_file_info_by_media_type("video")
    file_paths = [f['path'] for f in video_files if f.get('path')]
    
    logger.info(f"Found {len(file_paths)} video files: {file_paths}")
    
    result = await send_message_to_agent(
        agent_type="video", 
        message=query, 
        original_user_query=original_user_query,
        file_paths=file_paths
    )
    
    if result["status"] == "success":
        logger.info(f"Video processing successful: {len(result['result'])} characters returned")
        return {"video_results": result["result"]}
    else:
        logger.error(f"Video processing failed: {result['error_message']}")
        return {"error": f"Video processing failed: {result['error_message']}"}

async def image_modification_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """
    Process images and handle image-related tasks with direct path support.
    Now supports both file paths and URLs for maximum flexibility.
    """
    logger.info(f"Image task: '{query}'")
    
    # Get image file paths from global context
    image_files = get_file_info_by_media_type("image")
    file_paths = [f['path'] for f in image_files if f.get('path')]
    
    logger.info(f"Found {len(file_paths)} image files: {file_paths}")
    
    # Check if the query contains URLs or file paths and extract them
    file_urls = []
    clean_query = query
    
    if "http://" in query or "https://" in query or query.strip().startswith("C:"):
        # Extract potential file paths or URLs from the query
        import re
        # Match file paths (Windows/Unix) and URLs
        path_pattern = r'(?:[a-zA-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*|/(?:[^/\s]+/)*[^/\s]+|https?://[^\s]+)'
        potential_paths = re.findall(path_pattern, query)
        
        # Separate paths/URLs from the actual prompt
        for path in potential_paths:
            if path.startswith(('http://', 'https://')):
                file_urls.append(path)
                clean_query = clean_query.replace(path, '').strip()
            elif path.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff')):
                file_paths.append(path)
                clean_query = clean_query.replace(path, '').strip()
        
        logger.info(f"Extracted URLs from query: {file_urls}")
        logger.info(f"Extracted paths from query: {file_paths}")
        logger.info(f"Clean query after path extraction: '{clean_query}'")
    
    # If clean_query is empty or too short, use original query or provide default
    if not clean_query or len(clean_query.strip()) < 3:
        # Try to use original_user_query if it's different and longer
        if original_user_query and original_user_query != query and len(original_user_query.strip()) > 3:
            clean_query = original_user_query
            logger.info(f"Using original_user_query as clean_query: '{clean_query}'")
        else:
            # Provide a meaningful default based on context
            if any(word in query.lower() for word in ['transform', 'modify', 'change', 'edit']):
                clean_query = "Transform this image"
            elif any(word in query.lower() for word in ['enhance', 'improve', 'better']):
                clean_query = "Enhance this image"
            elif any(word in query.lower() for word in ['add', 'include', 'put']):
                clean_query = "Add elements to this image"
            else:
                clean_query = "Process this image"
            logger.info(f"Using default clean_query: '{clean_query}'")
    
    # Create a more structured message for the agent
    structured_message = clean_query
    
    # Add file information in a clear format
    if file_paths or file_urls:
        structured_message += "\n\n[IMAGE FILES TO PROCESS]"
        
        if file_paths:
            for i, path in enumerate(file_paths, 1):
                structured_message += f"\n{i}. File path: {path}"
        
        if file_urls:
            for i, url in enumerate(file_urls, len(file_paths) + 1):
                structured_message += f"\n{i}. File URL: {url}"
        
        structured_message += "\n\nPlease process the above image(s) according to the request."
    
    logger.info(f"Final structured message: {structured_message}")
    
    result = await send_message_to_agent(
        agent_type="image", 
        message=structured_message,  # Send the structured message
        original_user_query=original_user_query,
        file_paths=file_paths,
        file_urls=file_urls
    )
    
    if result["status"] == "success":
        logger.info(f"Image processing successful: {len(result['result'])} characters returned")
        return {"image_results": result["result"]}
    else:
        logger.error(f"Image processing failed: {result['error_message']}")
        return {"error": f"Image processing failed: {result['error_message']}"}
    
    
async def report_and_content_generation_function(query: str, original_user_query: str) -> Dict[str, Any]:
    """Generate reports, documents, and structured content with file support"""
    logger.info(f"Report task: '{query}'")
    
    # Get all file paths from global context for report generation
    all_file_paths = get_file_paths_from_global_context()
    
    logger.info(f"Found {len(all_file_paths)} files for report generation: {all_file_paths}")
    
    result = await send_message_to_agent(
        agent_type="report", 
        message=query, 
        original_user_query=original_user_query,
        file_paths=all_file_paths
    )
    
    if result["status"] == "success":
        logger.info(f"Report generation successful: {len(result['result'])} characters returned")
        return {"report_results": result["result"]}
    else:
        logger.error(f"Report generation failed: {result['error_message']}")
        return {"error": f"Report generation failed: {result['error_message']}"}

# HELPER FUNCTIONS FOR URL/PATH HANDLING

def is_url(path_or_url: str) -> bool:
    """Check if a string is a URL"""
    return path_or_url.startswith(('http://', 'https://'))

def is_file_path(path_or_url: str) -> bool:
    """Check if a string is a file path"""
    import os
    return os.path.exists(path_or_url) or path_or_url.startswith(('/', 'C:', 'D:', 'E:'))

def extract_media_paths_from_query(query: str) -> Dict[str, List[str]]:
    """Extract file paths and URLs from a query string"""
    import re
    
    # Pattern to match file paths and URLs
    path_pattern = r'(?:[a-zA-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*|/(?:[^/\s]+/)*[^/\s]+|https?://[^\s]+)'
    potential_paths = re.findall(path_pattern, query)
    
    file_paths = []
    file_urls = []
    
    for path in potential_paths:
        if is_url(path):
            file_urls.append(path)
        elif is_file_path(path):
            file_paths.append(path)
    
    return {
        "file_paths": file_paths,
        "file_urls": file_urls
    }

# BACKWARDS COMPATIBILITY FUNCTIONS (DEPRECATED BUT MAINTAINED)

def inject_multimedia_context_for_agent(agent_type: str, multimedia_parts: Optional[List] = None):
    """
    DEPRECATED: This function is maintained for backwards compatibility.
    New implementation uses direct path passing instead of context injection.
    """
    logger.warning("inject_multimedia_context_for_agent is deprecated. Use direct path passing instead.")
    pass

# Global variable for backwards compatibility (no longer used)
_current_multimedia_context = None

async def send_message_to_agent_with_context(
    agent_type: str,
    message: str,
    original_user_query: str,
    timeout: float = REQUEST_TIMEOUT
) -> Dict[str, Any]:
    """
    DEPRECATED: Enhanced version that checks for global multimedia context.
    Maintained for backwards compatibility. Use send_message_to_agent directly.
    """
    logger.warning("send_message_to_agent_with_context is deprecated. Use send_message_to_agent directly.")
    return await send_message_to_agent(
        agent_type=agent_type,
        message=message,
        original_user_query=original_user_query,
        timeout=timeout
    )