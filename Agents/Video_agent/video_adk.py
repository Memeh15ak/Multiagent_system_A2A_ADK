import os
import asyncio
import logging
import time
import google.generativeai as genai
from typing import List, Dict, Optional
import re 

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types as adk_types

# --- Configuration ---
logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING) 

MODEL_FOR_ADK_AGENT = "gemini-1.5-flash" 
MODEL_FOR_ANALYSIS = "gemini-2.5-flash-preview-05-20"  

VALID_VIDEO_FORMATS = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.mpeg']


class VideoManager:
    """Manages video uploads, caching, and cleanup for the session."""

    def __init__(self):
        self.files: Dict[str, genai.File] = {}
        self.analysis_cache: Dict[str, str] = {}
        self.file_order: List[str] = []

    def upload_video(self, path: str):
        """Uploads a video file to the Gemini API and stores its reference."""
        if path in self.files:
            return self.files[path]

        if not os.path.exists(path):
            print(f"Error: File not found at '{path}'")
            return None

        if not any(path.lower().endswith(fmt) for fmt in VALID_VIDEO_FORMATS):
            print(f"Error: Invalid video format for '{path}'")
            return None

        try:
            print(f"Uploading: {os.path.basename(path)}... (This might take a while for large videos)")
            uploaded_file = genai.upload_file(path=path)
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(5) 
                uploaded_file = genai.get_file(name=uploaded_file.name)

            if uploaded_file.state.name == "FAILED":
                print(f"Upload failed for: {os.path.basename(path)}. Error: {uploaded_file.state.error}")
                return None

            self.files[path] = uploaded_file
            if path not in self.file_order:
                self.file_order.append(path)
            print(f"Uploaded: {os.path.basename(path)}")
            return uploaded_file

        except Exception as e:
            print(f"An error occurred while uploading {os.path.basename(path)}: {e}")
            return None

    def get_videos_for_query(self, query: str) -> List[str]:
        """Intelligently selects videos based on filenames, ordinals, or context."""
        available = self.file_order
        if not available:
            return []

        query_lower = query.lower()

        mentioned_files = []
        for fmt in VALID_VIDEO_FORMATS:
            pattern = re.compile(rf"['\"]?([^'\"\s]+\{fmt})['\"]?", re.IGNORECASE)
            for match in pattern.findall(query_lower):
                found_path = next((p for p in available if os.path.basename(p).lower() == match.lower()), None)
                if found_path and found_path not in mentioned_files:
                    mentioned_files.append(found_path)
        
        if mentioned_files:
            return mentioned_files

        ordinal_map = {
            'first': 1, '1st': 1, 'one': 1, 'second': 2, '2nd': 2, 'two': 2,
            'third': 3, '3rd': 3, 'three': 3, 'fourth': 4, '4th': 4, 'four': 4,
            'fifth': 5, '5th': 5, 'five': 5, 'sixth': 6, '6th': 6, 'six': 6,
            'seventh': 7, '7th': 7, 'seven': 7, 'eighth': 8, '8th': 8, 'eight': 8,
            'ninth': 9, '9th': 9, 'nine': 9, 'tenth': 10, '10th': 10, 'ten': 10,
            'last': -1, 'final': -1
        }
        found_ordinals = []
        for word in query_lower.split():
            if word in ordinal_map:
                index = ordinal_map[word]
                file_index = index - 1 if index > 0 else -1
                if 0 <= file_index < len(available) or file_index == -1: 
                     found_ordinals.append(available[file_index])
        if found_ordinals:
            return list(dict.fromkeys(found_ordinals)) 

        return available

    def get_status(self) -> str:
        """Returns a string describing the current state of uploaded videos."""
        if not self.file_order:
            return "No videos have been uploaded yet."
        names = [os.path.basename(p) for p in self.file_order]
        return f"Available videos ({len(names)}): {', '.join(names)}"

    def cleanup(self):
        """Deletes all uploaded files from the API and clears all local caches."""
        if self.files:
            print("Cleaning up uploaded video files from Gemini API...")
            for uploaded_file in self.files.values():
                try:
                    genai.delete_file(name=uploaded_file.name)
                except Exception as e:
                    logger.error(f"Could not delete file {uploaded_file.name}: {e}")
            self.files.clear()
            self.analysis_cache.clear()
            self.file_order.clear()
            print("Video file cleanup complete!")


video_manager = VideoManager()



def upload_videos(file_paths: List[str]) -> Dict:
    """Tool to upload one or more video files."""
    if not file_paths:
        return {"status": "error", "message": "No file paths were provided for upload."}
    
    uploaded_count = 0
    for path in file_paths:
        if video_manager.upload_video(path):
            uploaded_count += 1

    if uploaded_count > 0:
        return {"status": "success", "message": f"Successfully processed {uploaded_count}/{len(file_paths)} video files."}
    else:
        return {"status": "error", "message": "No video files were successfully uploaded."}


def analyze_videos(query: str, file_paths: Optional[List[str]] = None) -> Dict:
    """
    Tool to analyze videos to answer a query. `file_paths` is optional.
    If `file_paths` is not provided, the tool will try to infer relevant videos
    from the `query` or use all available uploaded videos.
    """
    if file_paths is not None and len(file_paths) > 0:
        paths_to_analyze = file_paths
    else:
        paths_to_analyze = video_manager.get_videos_for_query(query)
        if not paths_to_analyze: 
             if not video_manager.files:
                return {"status": "error", "message": "Cannot analyze: No videos have been uploaded yet, and no specific video was mentioned."}
             else: 
                paths_to_analyze = video_manager.file_order


    if not paths_to_analyze:
        return {"status": "error", "message": f"Could not find relevant videos for your query. {video_manager.get_status()}"}

    # Generate a cache key that includes the query and the sorted base filenames
    cache_key = f"{query.lower().strip()}|{','.join(sorted([os.path.basename(p) for p in paths_to_analyze]))}"
    if cache_key in video_manager.analysis_cache:
        return {"status": "success", "analysis": video_manager.analysis_cache[cache_key]}

    uploaded_files_for_model = []
    # Ensure all specified paths are known to the VideoManager and get their genai.File objects
    for path in paths_to_analyze:
        uploaded_file = video_manager.upload_video(path) # This ensures the file is uploaded if not already
        if uploaded_file:
            uploaded_files_for_model.append(uploaded_file)
    
    if not uploaded_files_for_model:
        return {"status": "error", "message": "None of the specified videos are valid or available for analysis after attempted upload."}

    try:
        print(f"Performing new analysis on {len(uploaded_files_for_model)} video(s)...")

        prompt = [
            (
                "You are a meticulous video analyst. To answer the user's query, "
                "first, perform a step-by-step reasoning process to identify all relevant information from the video(s). "
                "Second, use this reasoning to formulate the final answer to the user's specific question. "
                "Provide only the direct and concise final answer. "
                f"User Query: '{query}'"
            )
        ] + uploaded_files_for_model

        model = genai.GenerativeModel(MODEL_FOR_ANALYSIS)
        response = model.generate_content(prompt)
        analysis_result = response.text.strip()
        video_manager.analysis_cache[cache_key] = analysis_result
        return {"status": "success", "analysis": analysis_result}

    except Exception as e:
        logger.error(f"Video analysis API call failed: {e}")
        return {"status": "error", "message": f"Analysis failed due to an API error: {str(e)}"}



memory = [
    "Previously, `concert_highlights.mp4` was uploaded and analyzed. It featured crowd cheering and stage lighting changes.",
    "The user once asked for a summary of `lecture_series_part1.mov`, focusing on key concepts presented.",
    "The last video analyzed was `traffic_cam_footage.mp4`, and the user was interested in vehicle types.",
    "User preference: When summarizing videos, include timestamps if possible.",
    "User preference: Highlight any unexpected events or anomalies in video analysis.",
    "I enjoy learning about ancient history." 
]


create_video_agent = Agent(
    name="video_analyzer_agent",
    model=MODEL_FOR_ADK_AGENT, 
    instruction=f"""You are a systematic video analyst agent. You will follow a strict step-by-step process to respond to user queries about videos.

    **Your Thought Process:**

    **Step 1: Identify User Intent, Keywords & Consult Memory for Video Context.**
    - Read the user's query carefully.
    - Identify action keywords (e.g., "analyze", "summarize", "describe", "compare", "identify", "what about").
    - Identify subject keywords:
        - Explicit video filenames (e.g., `my_clip.mp4`, `lecture.mov`). Look for patterns like `'.mp4'`, `'.mov'`, etc.
        - Ordinal references (e.g., `first video`, `second one`, `last clip`).
        - Implicit references to known videos (e.g., "the concert highlights", "the lecture video", "that traffic footage").
    - **Consult `Memory Context` for Video-Specific Information:**
        - **Relevant Recall:** If the user's query implicitly refers to a previously processed video (e.g., "what about the concert?", "summarize the last one", "tell me about the lecture video"), scan `Memory Context` for entries containing keywords that match the implied subject.
            - If a strong match is found (e.g., "the concert" implies `concert_highlights.mp4`, "the last one" implies `traffic_cam_footage.mp4`), *internally* resolve the implied subject to the explicit filename found in memory. This resolved filename will be used in subsequent steps if no new files are provided.
        - **User Preferences:** Note any user preferences from `Memory Context` that are directly related to *video analysis* (e.g., "include timestamps", "highlight anomalies") and keep them in mind for shaping your analysis output style.
        - **Filtering Irrelevant Memory:** Completely ignore and do not act upon any information in `Memory Context` that is not directly related to video analysis or user preferences for video analysis (e.g., "I enjoy learning about ancient history" should be disregarded for video analysis tasks).

    **Step 2: Formulate a Tool-Use Plan.**
    - **Crucial:** Always check if the prompt explicitly lists "video files located at the following paths: '[path1]', '[path2]'". This indicates *new* files provided by the A2A executor from the user's upload.
    - **Plan A (Upload & Analyze Explicit New Files):** If the prompt explicitly mentions "video files located at the following paths: ...", my plan is:
        1. Extract all mentioned video file paths from the prompt.
        2. Call `upload_videos` with all the extracted file paths.
        3. Call `analyze_videos` with the original user's full query (extracted from "User request:") and the *same* extracted file paths.
    - **Plan B (Analyze Existing/Implicit Files or Follow-up):** If no explicit *new* file paths are mentioned in the *current* prompt:
        1. If Step 1 successfully identified an implicit reference to a previously processed video from `Memory Context` (e.g., "the concert" resolved to `concert_highlights.mp4`), I MUST call `analyze_videos` using the resolved filename in the `query` parameter (see Step 3, Case 2).
        2. If no new explicit files, and no strong implicit reference from memory, then it's a follow-up on previously "uploaded" files (from the perspective of a `VideoManager` tool) or a general query. In this case, call `analyze_videos` with ONLY the original user's full query.

    **Step 3: Determine Tool Parameters.**
    - **Case 1 (Explicit New Filenames from A2A Executor):** If the prompt contains the specific phrase "video files located at the following paths: '[path1]', '[path2]'...", I MUST extract ALL these full paths (enclosed in single quotes) and pass them as a list to BOTH the `upload_videos` and `analyze_videos` tools' `file_paths` parameter. The user query for `analyze_videos` should be extracted from the part of the prompt that starts with "User request:".
    - **Case 2 (Implicit Context/Follow-up on Known Video via Memory):** If the user's query does NOT contain new filenames, but Step 1 resolved the query to a previously processed video (e.g., "what about the concert?" resolved to `concert_highlights.mp4`, or "summarize the last one" resolved to `traffic_cam_footage.mp4`) using `Memory Context`:
        - I MUST call `analyze_videos` with ONLY the `query` parameter.
        - The `file_paths` parameter MUST NOT be used here.
        - The `query` parameter for the tool MUST be carefully constructed to combine the user's original request AND the specific filename of the implied video from `Memory Context` (e.g., if the user asked "What about the concert?", the tool call becomes `analyze_videos(query="What about `concert_highlights.mp4`?")`). This is crucial as it allows the `analyze_videos` tool to correctly identify and process the specific video from its internal storage or cache.
    - **Case 3 (General Follow-up / No New Files, No Implicit Memory Match):** If no new explicit files and no strong implicit reference from memory, I MUST call `analyze_videos` with ONLY the `query` parameter (the original user query). I will NOT use the `file_paths` parameter in this case. The `analyze_videos` tool is expected to handle this by perhaps analyzing the *last* uploaded video if relevant, or providing a general analysis.

    **Step 4: Execute and Report.**
    - Call the tool(s) according to the plan.
    - Take the `analysis` text from the final tool output and present it as my final answer, without any modification or added text. Ensure any user preferences noted from `Memory Context` (e.g., "include timestamps") are reflected in the final output style.

    ---

    **Memory Context:** {memory}
    """
    ,
    tools=[upload_videos, analyze_videos] 
)