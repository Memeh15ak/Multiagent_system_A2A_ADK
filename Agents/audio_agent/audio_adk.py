import os
import logging
import time
import google.generativeai as genai
from google.adk.agents import LlmAgent
from dotenv import load_dotenv
from google.adk.agents import Agent

load_dotenv()
logger = logging.getLogger(__name__)

MODEL_GEMINI_AUDIO = "gemini-2.0-flash-exp"

def analyze_audio_query(audio_file_path: str, user_query: str) -> dict:
    """
    Analyze audio file using Gemini Files API and answer user query
    
    Args:
        audio_file_path: Path to the audio file
        user_query: User's specific question/request about the audio
        
    Returns:
        dict with analysis results
    """
    try:
        logger.info(f"Processing audio query: {user_query}")
        logger.info(f"Audio file: {audio_file_path}")
        
        if not os.path.exists(audio_file_path):
            return {
                "status": "failed",
                "error": "Audio file not found",
                "response": f"Error: Audio file not found at path: {audio_file_path}"
            }
    
        uploaded_file = genai.upload_file(
            path=audio_file_path, 
            display_name=os.path.basename(audio_file_path)
        )

        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
        
        if uploaded_file.state.name == "FAILED":
            return {
                "status": "failed",
                "error": "Audio file processing failed",
                "response": "Error: Audio file processing failed on Gemini API"
            }
        
        # Create enhanced prompt for user query
        prompt = f"""
        You are an expert audio analyst. Please analyze this audio file and provide a detailed response to the user's query.

        User Query: "{user_query}"

        Instructions:
        1. Listen to the entire audio file carefully
        2. Provide a direct, comprehensive answer to the user's specific question
        3. Include relevant details about what you hear (voices, sounds, environment, etc.)
        4. If the query is general (like "What can be heard"), provide a complete description
        5. Be specific and detailed in your response
        6. Format your response clearly and conversationally

        Please analyze the audio and respond directly to the user's query.
        """
        
        # Generate response
        model = genai.GenerativeModel(MODEL_GEMINI_AUDIO)
        response = model.generate_content([prompt, uploaded_file])
        
        # Cleanup uploaded file
        try:
            genai.delete_file(uploaded_file.name)
            logger.info("Successfully cleaned up uploaded file")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
        
        if response.text:
            result_text = response.text.strip()
            logger.info(f"Generated response with {len(result_text)} characters")
            
            return {
                "status": "success",
                "response": result_text,
                "audio_results": result_text,  # For compatibility
                "query": user_query,
                "file_processed": audio_file_path,
                "summary": f"Successfully analyzed audio file and answered: {user_query}"
            }
        else:
            return {
                "status": "failed",
                "error": "No response from Gemini",
                "response": "Error: No response generated from the audio analysis"
            }
            
    except Exception as e:
        error_msg = f"Audio analysis error: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": str(e),
            "response": f"Error analyzing audio: {str(e)}"
        }

def create_audio_agent():
    """Create audio processing agent using Gemini Files API - FIXED VERSION"""
    
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=gemini_api_key)
        
        logger.info(f"Creating audio agent with model: {MODEL_GEMINI_AUDIO}")
        
        tools = [analyze_audio_query]
        
        audio_agent = Agent(
            name='audio_agent_gemini',
            model=MODEL_GEMINI_AUDIO,
            description='Audio processing agent that analyzes audio files and answers user queries using Gemini Files API',
            instruction="""You are an expert audio processing assistant. Your role is to analyze audio files and provide detailed, helpful responses to user queries.

When you receive a request to analyze an audio file:

1. ALWAYS call the analyze_audio_query function with the provided file path and user query
2. Extract the file path from the user's message (look for file paths in the format like "C:\\path\\to\\file.mp3" or "/path/to/file.mp3")
3. Use the user's specific question as the query parameter
4. Return the complete analysis result directly to the user

Key points:
- Always process the actual audio file, never try to answer without calling the function
- Look for file paths in the user's message, they may be provided as "File path: /path/to/audio.mp3"
- If multiple file paths are provided, process the first audio file you find
- Provide comprehensive, detailed responses based on what you actually hear in the audio
- Be conversational and helpful in your responses

Example format for calling the function:
analyze_audio_query(audio_file_path="/path/to/audio.mp3", user_query="What can be heard in this file")

Always call the function first, then return the results to the user.""",
            tools=tools,
        )
        
        logger.info("✅ Audio agent created successfully")
        return audio_agent, audio_agent
        
    except Exception as e:
        logger.error(f"Error creating audio agent: {e}")
        raise RuntimeError(f"Failed to create audio agent: {e}")
    
async def create_audio_agent_async() -> Agent:
    """Create audio processing agent using Gemini Files API - Original signature"""
    
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=gemini_api_key)
        
        logger.info(f"Creating audio agent with model: {MODEL_GEMINI_AUDIO}")
        
        tools = [analyze_audio_query]
        
        audio_agent = Agent(
            name='audio_agent_gemini',
            model=MODEL_GEMINI_AUDIO,
            description='Audio processing agent that analyzes audio files and answers user queries using Gemini Files API',
            instruction="""You are an expert audio processing assistant. Your role is to analyze audio files and provide detailed, helpful responses to user queries.

When you receive a request to analyze an audio file:

1. ALWAYS call the analyze_audio_query function with the provided file path and user query
2. Extract the file path from the user's message (look for file paths in the format like "C:\\path\\to\\file.mp3" or "/path/to/file.mp3")
3. Use the user's specific question as the query parameter
4. Return the complete analysis result directly to the user

Key points:
- Always process the actual audio file, never try to answer without calling the function
- Look for file paths in the user's message, they may be provided as "File path: /path/to/audio.mp3"
- If multiple file paths are provided, process the first audio file you find
- Provide comprehensive, detailed responses based on what you actually hear in the audio
- Be conversational and helpful in your responses

Example format for calling the function:
analyze_audio_query(audio_file_path="/path/to/audio.mp3", user_query="What can be heard in this file")

Always call the function first, then return the results to the user.""",
            tools=tools,
        )
        return audio_agent
    except Exception as e:
        logger.error(f"Error creating audio agent: {e}")
        raise RuntimeError(f"Failed to create audio agent: {e}")