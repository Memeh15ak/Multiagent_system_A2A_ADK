import streamlit as st
import asyncio
import httpx
import json
import logging
import time
from typing import Dict, Any, Optional, List
from uuid import uuid4
import os
from datetime import datetime
import base64
from io import BytesIO
import tempfile
import shutil
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_ORCHESTRATOR_URL = "http://localhost:10020"
REQUEST_TIMEOUT = 120.0
POLL_INTERVAL = 2.0
MAX_POLL_ATTEMPTS = 60

# Agent information for display
AGENT_INFO = {
    "web_search": {"name": "Web Search", "icon": "🔍", "color": "#1f77b4"},
    "code": {"name": "Code Executor", "icon": "💻", "color": "#ff7f0e"},
    "audio": {"name": "Audio Processor", "icon": "🎵", "color": "#2ca02c"},
    "image": {"name": "Image Editor", "icon": "🎨", "color": "#d62728"},
    "report": {"name": "Report Generator", "icon": "📊", "color": "#9467bd"},
    "excel": {"name": "Excel Analyzer", "icon": "📈", "color": "#8c564b"},
    "rag": {"name": "Document RAG", "icon": "📚", "color": "#e377c2"},
    "image_analyzer": {"name": "Image Analyzer", "icon": "🔬", "color": "#7f7f7f"},
    "image_generation": {"name": "Image Generator", "icon": "🎯", "color": "#bcbd22"},
    "video": {"name": "Video Processor", "icon": "🎬", "color": "#17becf"}
}

def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'orchestrator_url' not in st.session_state:
        st.session_state.orchestrator_url = DEFAULT_ORCHESTRATOR_URL
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = {}
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'temp_file_paths' not in st.session_state:
        st.session_state.temp_file_paths = []

async def check_orchestrator_status(url: str) -> Dict[str, Any]:
    """Check if orchestrator is available using A2A protocol"""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            # Try to get agent card (A2A standard endpoint)
            try:
                card_response = await client.get(f"{url}/")
                if card_response.status_code == 200:
                    card_data = card_response.json()
                    return {
                        "available": True,
                        "card": card_data,
                        "skills_count": len(card_data.get("skills", [])),
                        "name": card_data.get("name", "ADK Orchestrator"),
                        "description": card_data.get("description", ""),
                        "version": card_data.get("version", "unknown")
                    }
            except Exception as e:
                logger.warning(f"Agent card request failed: {e}")
            
            # Fallback: try a simple GET request to root to check if server is responding
            try:
                root_response = await client.get(url)
                if root_response.status_code in [200, 405]:  # 405 is "Method Not Allowed" but means server is up
                    return {"available": True, "name": "ADK Orchestrator", "note": "Server responding"}
            except Exception as e:
                logger.warning(f"Root request failed: {e}")
            
            return {"available": False, "error": "No valid endpoints found"}
    except Exception as e:
        return {"available": False, "error": str(e)}

def save_uploaded_files_to_temp(files: List) -> List[str]:
    """Save uploaded files to temporary directory and return file paths"""
    temp_paths = []
    
    if files:
        # Create temp directory for this session
        temp_dir = tempfile.mkdtemp(prefix="streamlit_files_")
        
        for file in files:
            try:
                # Create temp file path
                temp_file_path = os.path.join(temp_dir, file.name)
                
                # Write file content to temp location
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(file.read())
                
                # Reset file pointer for potential re-use in UI
                file.seek(0)
                
                temp_paths.append(temp_file_path)
                logger.info(f"Saved uploaded file to: {temp_file_path}")
                
            except Exception as e:
                logger.error(f"Error saving file {file.name}: {e}")
    
    return temp_paths

def get_mime_type(file_name: str) -> str:
    """Get MIME type based on file extension"""
    ext = file_name.lower().split('.')[-1]
    mime_types = {
        'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
        'gif': 'image/gif', 'bmp': 'image/bmp', 'webp': 'image/webp',
        'mp3': 'audio/mpeg', 'wav': 'audio/wav', 'mp4': 'video/mp4',
        'pdf': 'application/pdf', 'txt': 'text/plain',
        'csv': 'text/csv', 'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    }
    return mime_types.get(ext, 'application/octet-stream')

def create_a2a_message_payload(message_text: str, files: Optional[List] = None):
    """Create A2A protocol message payload with file paths (like CLI client)"""
    # Create text part
    parts = [{"kind": "text", "text": message_text}]
    
    # Save files to temp and add file parts with file paths
    if files:
        temp_file_paths = save_uploaded_files_to_temp(files)
        st.session_state.temp_file_paths.extend(temp_file_paths)  # Track for cleanup
        
        for temp_path in temp_file_paths:
            try:
                mime_type = get_mime_type(os.path.basename(temp_path))
                
                # Create file part with absolute path (like CLI client)
                parts.append({
                    "kind": "file",
                    "file": {
                        "uri": os.path.abspath(temp_path),  # Absolute file path
                        "mime_type": mime_type
                    }
                })
                logger.info(f"Added file part: {temp_path} ({mime_type})")
                
            except Exception as e:
                logger.error(f"Error creating file part for {temp_path}: {e}")
    
    # Create A2A compliant payload matching CLI client
    return {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": parts,
                "messageId": uuid4().hex,
            },
            "agentId": "research_orchestrator",
            "userId": "streamlit_user",
        },
        "id": uuid4().hex
    }

async def send_message_to_orchestrator(url: str, message: str, files: Optional[List] = None) -> Dict[str, Any]:
    """Send message to orchestrator using A2A protocol"""
    try:
        payload = create_a2a_message_payload(message, files)
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
            # Use A2A JSON-RPC endpoint (standard A2A protocol)
            response = await client.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle A2A JSON-RPC response
                if "result" in result:
                    a2a_result = result["result"]
                    
                    # Check if it's a task-based response
                    if isinstance(a2a_result, dict) and "id" in a2a_result and "status" in a2a_result:
                        # It's a task - poll for completion
                        task_id = a2a_result["id"]
                        return await poll_task_completion_a2a(client, url, task_id)
                    
                    # Direct message response
                    return {"status": "success", "result": a2a_result}
                
                # Handle error response
                if "error" in result:
                    return {"status": "error", "error": result["error"]["message"]}
                
                # Fallback
                return {"status": "success", "result": result}
            else:
                error_text = ""
                try:
                    error_data = response.json()
                    error_text = error_data.get('error', response.text)
                except:
                    error_text = response.text
                return {"status": "error", "error": f"HTTP {response.status_code}: {error_text}"}
                
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return {"status": "error", "error": str(e)}

async def poll_task_completion_a2a(client: httpx.AsyncClient, url: str, task_id: str) -> Dict[str, Any]:
    """Poll task until completion using A2A protocol"""
    for attempt in range(MAX_POLL_ATTEMPTS):
        try:
            # Create A2A get_task request with correct method name
            get_task_payload = {
                "jsonrpc": "2.0",
                "method": "tasks/get",  # Correct A2A method name
                "params": {"id": task_id},
                "id": uuid4().hex
            }
            
            response = await client.post(url, json=get_task_payload)
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    task = result["result"]
                    state = task.get('status', {}).get('state', 'unknown')
                    
                    if state in ['completed', 'failed', 'canceled']:
                        return {"status": "success", "result": task}
            
            await asyncio.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.warning(f"Polling error: {e}")
    
    return {"status": "error", "error": "Task polling timed out"}

def find_generated_files(response_text: str) -> List[str]:
    """Extract generated file names from response text"""
    generated_files = []
    
    # Look for common patterns of generated files
    patterns = [
        r'Generated image saved as: ([^\s]+\.(?:png|jpg|jpeg|gif|bmp|webp))',
        r'saved as: ([^\s]+\.(?:png|jpg|jpeg|gif|bmp|webp))',
        r'output file: ([^\s]+\.(?:png|jpg|jpeg|gif|bmp|webp))',
        r'generated_([a-f0-9]+\.(?:png|jpg|jpeg|gif|bmp|webp))',
        r'(generated_[a-f0-9]+\.(?:png|jpg|jpeg|gif|bmp|webp))',  # More specific pattern
        r'([^\s]*generated_[^/\s]*\.(?:png|jpg|jpeg|gif|bmp|webp))',  # Path with generated_
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        for match in matches:
            # Clean up the match - remove any leading/trailing quotes or spaces
            clean_match = match.strip().strip('"\'')
            if clean_match and clean_match not in generated_files:
                generated_files.append(clean_match)
    
    return generated_files

def display_generated_images(response_text: str):
    """Display generated images if found in response"""
    generated_files = find_generated_files(response_text)
    
    if generated_files:
        st.subheader("🎨 Generated Images")
        
        for file_name in generated_files:
            # Get the base filename without any path
            base_filename = os.path.basename(file_name)
            
            # Try different possible locations for the generated file
            possible_paths = [
                file_name,  # Direct path as provided
                base_filename,  # Just the filename in current directory
                os.path.join(os.getcwd(), base_filename),  # Current working directory
                os.path.join(os.path.dirname(os.path.abspath(__file__)), base_filename),  # Script directory
                os.path.join(os.path.expanduser("~"), base_filename),  # Home directory
                os.path.join("/tmp", base_filename),  # Common temp location
                os.path.join(tempfile.gettempdir(), base_filename),  # System temp
                # Also try looking in subdirectories
                os.path.join(os.getcwd(), "generated", base_filename),
                os.path.join(os.getcwd(), "images", base_filename),
                os.path.join(os.getcwd(), "output", base_filename),
            ]
            
            # Add any uploaded file temp directories to search paths
            if hasattr(st.session_state, 'temp_file_paths') and st.session_state.temp_file_paths:
                for temp_path in st.session_state.temp_file_paths:
                    temp_dir = os.path.dirname(temp_path)
                    possible_paths.append(os.path.join(temp_dir, base_filename))
            
            found_file = None
            for path in possible_paths:
                try:
                    if os.path.exists(path) and os.path.isfile(path):
                        found_file = path
                        break
                except Exception as e:
                    # Skip invalid paths
                    continue
            
            if found_file:
                try:
                    # Display the image
                    st.image(found_file, caption=f"Generated: {base_filename}", use_container_width=True)
                    
                    # Add download button
                    with open(found_file, "rb") as img_file:
                        img_bytes = img_file.read()
                        st.download_button(
                            label=f"📥 Download {base_filename}",
                            data=img_bytes,
                            file_name=base_filename,
                            mime=get_mime_type(base_filename),
                            key=f"download_{base_filename}_{uuid4().hex[:8]}"
                        )
                    
                    st.success(f"✅ Found and displayed: {base_filename}")
                    st.info(f"📍 File location: {found_file}")
                    
                except Exception as e:
                    st.error(f"❌ Error displaying {base_filename}: {e}")
            else:
                st.warning(f"⚠️ Generated file not found: {base_filename}")
                
                # Show a more detailed search info
                with st.expander(f"🔍 Search details for {base_filename}"):
                    st.text("Searched in these locations:")
                    for i, path in enumerate(possible_paths[:10]):  # Show first 10 paths
                        exists = "✅" if os.path.exists(path) else "❌"
                        st.text(f"  {exists} {path}")
                    if len(possible_paths) > 10:
                        st.text(f"  ... and {len(possible_paths) - 10} more locations")
                    
                    # Also show current working directory contents for debugging
                    try:
                        cwd_files = [f for f in os.listdir(os.getcwd()) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]
                        if cwd_files:
                            st.text("\n📁 Image files in current directory:")
                            for f in cwd_files:
                                st.text(f"  • {f}")
                    except Exception as e:
                        st.text(f"Could not list current directory: {e}")

def display_generated_content(response_text: str):
    """Display any generated content (images, files, etc.)"""
    # Check for generated images
    display_generated_images(response_text)
    
    # You can add more content types here later
    # display_generated_videos(response_text)
    # display_generated_documents(response_text)

def extract_response_text(result: Dict[str, Any]) -> str:
    """Extract readable text from A2A orchestrator response"""
    if not result or result.get("status") != "success":
        return result.get("error", "Unknown error occurred")
    
    data = result.get("result", {})
    
    # Handle task result (your CLI client pattern)
    if "artifacts" in data:
        artifacts = data["artifacts"]
        if artifacts and len(artifacts) > 0:
            for artifact in artifacts:
                parts = artifact.get("parts", [])
                if parts and len(parts) > 0:
                    for part in parts:
                        if "text" in part:
                            return part["text"]
    
    # Handle status message with parts
    if "status" in data and "message" in data["status"]:
        message = data["status"]["message"]
        if "parts" in message:
            for part in message["parts"]:
                if "text" in part:
                    return part["text"]
    
    # Handle direct message response
    if "parts" in data:
        for part in data["parts"]:
            if "text" in part:
                return part["text"]
    
    # Handle simple response
    if "response" in data:
        return data["response"]
    
    # Fallback
    return json.dumps(data, indent=2)

def render_sidebar():
    """Render sidebar with configuration and status"""
    with st.sidebar:
        st.header("🎛️ Configuration")
        
        # Orchestrator URL configuration
        new_url = st.text_input(
            "Orchestrator URL",
            value=st.session_state.orchestrator_url,
            help="URL of your ADK Translation Orchestrator Agent"
        )
        
        if new_url != st.session_state.orchestrator_url:
            st.session_state.orchestrator_url = new_url
            st.rerun()
        
        # Check status button
        if st.button("🔄 Check Status", use_container_width=True):
            with st.spinner("Checking orchestrator status..."):
                status = asyncio.run(check_orchestrator_status(st.session_state.orchestrator_url))
                st.session_state.agent_status = status
                st.rerun()
        
        # Display status
        st.subheader("📊 Orchestrator Status")
        if st.session_state.agent_status:
            status = st.session_state.agent_status
            if status.get("available"):
                st.success(f"✅ Online - {status.get('name', 'ADK Orchestrator')}")
                if "skills_count" in status:
                    st.info(f"🎯 {status['skills_count']} skills available")
                if "version" in status:
                    st.caption(f"Version: {status['version']}")
                if "description" in status:
                    with st.expander("ℹ️ Description"):
                        st.text(status['description'])
            else:
                st.error(f"❌ Offline - {status.get('error', 'Unknown error')}")
        else:
            st.warning("⚠️ Status unknown - Click 'Check Status'")
        
        # Agent capabilities info
        st.subheader("🤖 Available Agents")
        cols = st.columns(2)
        
        agents_list = list(AGENT_INFO.items())
        for i, (agent_id, info) in enumerate(agents_list):
            col = cols[i % 2]
            with col:
                st.markdown(f"**{info['icon']} {info['name']}**")
        
        # File upload section
        st.subheader("📁 File Upload")
        uploaded_files = st.file_uploader(
            "Upload files for processing",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'xlsx', 'csv', 'jpg', 'png', 'gif', 'mp3', 'wav', 'mp4'],
            help="Upload files that agents can process"
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"📎 {len(uploaded_files)} files uploaded")
            
            # Show uploaded files
            for file in uploaded_files:
                st.text(f"• {file.name} ({file.size} bytes)")

        # Connection test
        st.subheader("🔧 Quick Test")
        if st.button("Test Connection", use_container_width=True):
            with st.spinner("Testing connection..."):
                test_result = asyncio.run(send_message_to_orchestrator(
                    st.session_state.orchestrator_url,
                    "Hello! Please confirm you're working and list available agents."
                ))
                
                if test_result.get("status") == "success":
                    st.success("✅ Connection successful!")
                    response = extract_response_text(test_result)
                    st.info(response[:200] + "..." if len(response) > 200 else response)
                else:
                    st.error(f"❌ Connection failed: {test_result.get('error')}")

def render_example_queries():
    """Render example queries section"""
    st.subheader("💡 Example Queries")
    
    examples = {
        "Web Search & Analysis": [
            "Search for the latest AI developments and create a summary report",
            "Find information about Python best practices and generate code examples"
        ],
        "Data Analysis": [
            "Analyze the uploaded Excel file and create visualizations",
            "Generate insights from the CSV data with charts and graphs"
        ],
        "Content Generation": [
            "Create a comprehensive report about machine learning trends",
            "Generate a presentation outline for a tech conference"
        ],
        "Image Generation": [
            "Generate an image of a futuristic city skyline",
            "Create a logo for a tech startup",
            "Generate an abstract art piece with vibrant colors"
        ],
        "Multimedia Processing": [
            "Analyze the uploaded images and describe what you see",
            "Process the audio file and provide a transcript",
            "Extract information from the uploaded documents"
        ],
        "Code & Technical": [
            "Write a Python script to process CSV files",
            "Create a data visualization dashboard using the uploaded data",
            "Debug this code and suggest improvements"
        ]
    }
    
    cols = st.columns(2)
    for i, (category, queries) in enumerate(examples.items()):
        col = cols[i % 2]
        with col:
            with st.expander(f"📋 {category}"):
                for query in queries:
                    if st.button(query, key=f"example_{hash(query)}", use_container_width=True):
                        st.session_state.example_query = query
                        st.rerun()

def render_chat_interface():
    """Render main chat interface"""
    st.header("💬 Chat with ADK Orchestrator")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("timestamp"):
                st.caption(f"⏰ {message['timestamp']}")

def cleanup_temp_files():
    """Clean up temporary files"""
    if hasattr(st.session_state, 'temp_file_paths'):
        for temp_path in st.session_state.temp_file_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    # Also try to remove the parent temp directory if empty
                    temp_dir = os.path.dirname(temp_path)
                    try:
                        os.rmdir(temp_dir)
                    except OSError:
                        pass  # Directory not empty, that's ok
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_path}: {e}")
        st.session_state.temp_file_paths = []

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="ADK Agent Orchestrator",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .agent-card {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }
    .status-online { color: #28a745; }
    .status-offline { color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 ADK Agent Orchestrator Interface</h1>
        <p>Interact with your multi-agent system through a user-friendly interface</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        render_chat_interface()
        
        # Handle example query selection
        if hasattr(st.session_state, 'example_query'):
            query = st.session_state.example_query
            del st.session_state.example_query
        else:
            query = ""
        
        # Chat input
        if user_input := st.chat_input("Ask the orchestrator anything...", key="main_input"):
            query = user_input
        
        # Process query
        if query:
            # Add user message
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append({
                "role": "user",
                "content": query,
                "timestamp": timestamp
            })
            
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(query)
                st.caption(f"⏰ {timestamp}")
            
            # Show thinking spinner and send to orchestrator
            with st.chat_message("assistant"):
                with st.spinner("🤖 Processing your request..."):
                    result = asyncio.run(send_message_to_orchestrator(
                        st.session_state.orchestrator_url,
                        query,
                        st.session_state.uploaded_files
                    ))
                
                # Extract and display response
                response_text = extract_response_text(result)
                st.markdown(response_text)
                
                # Display any generated images or files
                display_generated_content(response_text)
                
                response_timestamp = datetime.now().strftime("%H:%M:%S")
                st.caption(f"⏰ {response_timestamp}")
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": response_timestamp
                })
    
    with col2:
        # Example queries
        render_example_queries()
    
    # Cleanup on exit
    try:
        cleanup_temp_files()
    except:
        pass

if __name__ == "__main__":
    main()