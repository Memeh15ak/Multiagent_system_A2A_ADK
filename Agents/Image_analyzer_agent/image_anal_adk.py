import os
import asyncio
import logging
import time
import google.generativeai as genai
from typing import List, Dict, Optional

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types as adk_types

logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

MODEL = "gemini-1.5-flash" 
VALID_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']

class ImageManager:
    """Manages image uploads, caching, and cleanup for the session."""

    def __init__(self):
        self.files: Dict[str, genai.File] = {}
        self.analysis_cache: Dict[str, str] = {}
        self.file_order: List[str] = []

    def upload_image(self, path: str):
        if path in self.files:
            return self.files[path]

        if not os.path.exists(path):
            print(f"Error: File not found at '{path}'")
            return None

        if not any(path.lower().endswith(fmt) for fmt in VALID_IMAGE_FORMATS):
            print(f"Error: Invalid image format for '{path}'")
            return None

        try:
            print(f"Uploading: {os.path.basename(path)}...")
            uploaded_file = genai.upload_file(path=path)
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(1)
                uploaded_file = genai.get_file(name=uploaded_file.name)

            if uploaded_file.state.name == "FAILED":
                print(f"Upload failed for: {os.path.basename(path)}")
                return None

            self.files[path] = uploaded_file
            if path not in self.file_order:
                self.file_order.append(path)
            print(f"Uploaded: {os.path.basename(path)}")
            return uploaded_file

        except Exception as e:
            print(f"An error occurred while uploading {os.path.basename(path)}: {e}")
            return None

    def get_images_for_query(self, query: str) -> List[str]:
        """Intelligently selects images based on filenames, ordinals, or context."""
        available = self.file_order
        if not available:
            return []

        query_lower = query.lower()

        mentioned_files = [path for path in available if os.path.basename(path).lower() in query_lower]
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
        """Returns a string describing the current state of uploaded images."""
        if not self.file_order:
            return "No images have been uploaded yet."
        names = [os.path.basename(p) for p in self.file_order]
        return f"Available images ({len(names)}): {', '.join(names)}"

    def cleanup(self):
        """Deletes all uploaded files from the API and clears all local caches."""
        if self.files:
            print("Cleaning up uploaded files...")
            for uploaded_file in self.files.values():
                try:
                    genai.delete_file(name=uploaded_file.name)
                except Exception as e:
                    logger.error(f"Could not delete file {uploaded_file.name}: {e}")
            self.files.clear()
            self.analysis_cache.clear()
            self.file_order.clear()
            print("Cleanup complete!")

image_manager = ImageManager()



def upload_images(file_paths: List[str]) -> Dict:
    """Tool to upload one or more images."""
    if not file_paths:
        return {"status": "error", "message": "No file paths were provided."}
    uploaded_count = sum(1 for path in file_paths if image_manager.upload_image(path))
    if uploaded_count > 0:
        return {"status": "success", "message": f"Successfully processed {uploaded_count}/{len(file_paths)} files."}
    else:
        return {"status": "error", "message": "No images were successfully uploaded."}


def analyze_images(query: str, file_paths: Optional[List[str]] = None) -> Dict:
    """Tool to analyze images to answer a query. `file_paths` is optional."""
    if not image_manager.files and not file_paths: 
        return {"status": "error", "message": "Cannot analyze: No images have been uploaded yet, and no new paths were specified for upload."}

    paths_to_analyze = file_paths if file_paths is not None else image_manager.get_images_for_query(query)
    if not paths_to_analyze:
        return {"status": "error", "message": f"Could not find relevant images for your query. {image_manager.get_status()}"}

    cache_key = f"{query.lower().strip()}|{','.join(sorted([os.path.basename(p) for p in paths_to_analyze]))}"
    if cache_key in image_manager.analysis_cache:
        return {"status": "success", "analysis": image_manager.analysis_cache[cache_key]}

    uploaded_files_for_model = []
    for path in paths_to_analyze:
        uploaded_file = image_manager.upload_image(path)
        if uploaded_file:
            uploaded_files_for_model.append(uploaded_file)
    
    if not uploaded_files_for_model:
        return {"status": "error", "message": "None of the specified images are valid or available for analysis after attempted upload."}

    try:
        print(f"Performing new analysis on {len(uploaded_files_for_model)} image(s)...")

        prompt = [
            (
                "You are a meticulous visual analyst. To answer the user's query, "
                "first, perform a step-by-step reasoning process to identify all relevant objects. "
                "Second, use this reasoning to formulate the final answer to the user's specific question. "
                "Provide only the direct and concise final answer. "
                f"User Query: '{query}'"
            )
        ] + uploaded_files_for_model

        model = genai.GenerativeModel(MODEL)
        response = model.generate_content(prompt)
        analysis_result = response.text.strip()
        image_manager.analysis_cache[cache_key] = analysis_result
        return {"status": "success", "analysis": analysis_result}

    except Exception as e:
        logger.error(f"Analysis API call failed: {e}")
        return {"status": "error", "message": f"Analysis failed due to an API error: {str(e)}"}



memory = [
    "Previously, `dog_park.jpg` was uploaded and analyzed. It contained a Golden Retriever in a park.",
    "The user previously asked about the dominant color in `sunset_view.png`.",
    "The last image explicitly analyzed was `family_portrait.jpeg`, and the user wanted to know about facial expressions.",
    "User preference: When analyzing, prioritize details about objects over background if not specified.",
    "User preference: Provide concise descriptions unless more detail is explicitly requested.",
    "User's mood seems positive based on last interaction.", 
]


create_image_agent = Agent(
    name="image_analyzer_agent",
    model=LiteLlm(model=f"gemini/{MODEL}"), 
    instruction=f"""You are a systematic visual analyst agent. You will follow a strict step-by-step process to respond to user queries about images.

    **Your Thought Process:**

    **Step 1: Identify User Intent, Keywords & Consult Memory for Image Context.**
    - Read the user's query carefully.
    - Identify action keywords (e.g., "analyze", "compare", "count", "describe", "what about").
    - Identify subject keywords (e.g., explicit filenames like `image.jpg`, ordinals like `first`/`last`, general descriptions like "the dog picture").
    - **Consult `Memory Context` for Image-Specific Information:**
        - **Relevant Recall:** If the user's query implicitly refers to a previously processed image (e.g., "what about the dog?", "describe the last one", "tell me about the sunset picture"), scan `Memory Context` for entries containing keywords that match the implied subject.
            - If a strong match is found (e.g., "the dog" strongly implies `dog_park.jpg` from memory, "the last one" implies `family_portrait.jpeg`), *internally* resolve the implied subject to the explicit filename found in memory. This resolved filename will be used in subsequent steps if no new files are provided.
        - **User Preferences:** Note any user preferences from `Memory Context` that are directly related to *image analysis* (e.g., "concise answers", "prioritize objects") and keep them in mind for shaping your analysis output style.
        - **Filtering Irrelevant Memory:** Completely ignore and do not act upon any information in `Memory Context` that is not directly related to image analysis or user preferences for image analysis (e.g., "I like reading historical fiction" or "User's mood seems positive" should be disregarded for image analysis tasks).

    **Step 2: Check for provided file paths and formulate a Tool-Use Plan.**
    - **Crucial:** Always check if the prompt explicitly lists "image files located at the following paths: '[path1]', '[path2]'". This indicates *new* files provided by the A2A executor from the user's upload.
    - **Plan A (Upload & Analyze Explicit New Files):** If the prompt explicitly mentions "image files located at the following paths: ...", my plan is:
        1. Call `upload_images` with all the extracted file paths.
        2. Call `analyze_images` with the original user query and the *same* extracted file paths.
    - **Plan B (Analyze Existing/Implicit Files or Follow-up):** If no explicit *new* file paths are mentioned in the *current* prompt:
        1. If Step 1 successfully identified an implicit reference to a previously processed image from `Memory Context` (e.g., "the dog" resolved to `dog_park.jpg`), I MUST call `analyze_images` using the resolved filename in the `query` parameter (see Step 3, Case 2).
        2. If no new explicit files, and no strong implicit reference from memory, then it's a follow-up on previously "uploaded" files (from the perspective of an `ImageManager` tool) or a general query. In this case, call `analyze_images` with ONLY the original user query.

    **Step 3: Determine Tool Parameters.**
    - **Case 1 (Explicit New Filenames from A2A Executor):** If the prompt contains the specific phrase "image files located at the following paths: '[path1]', '[path2]'...", I MUST extract ALL these full paths (enclosed in single quotes) and pass them as a list to BOTH the `upload_images` and `analyze_images` tools' `file_paths` parameter. The user query for `analyze_images` should be extracted from the part of the prompt that starts with "User request:".
    - **Case 2 (Implicit Context/Follow-up on Known Image via Memory):** If the user's query does NOT contain new filenames, but Step 1 resolved the query to a previously processed image (e.g., "what about the dog?" resolved to `dog_park.jpg`, or "describe the last one" resolved to `family_portrait.jpeg`) using `Memory Context`:
        - I MUST call `analyze_images` with ONLY the `query` parameter.
        - The `file_paths` parameter MUST NOT be used here.
        - The `query` parameter for the tool MUST be carefully constructed to combine the user's original request AND the specific filename of the implied image from `Memory Context` (e.g., if the user asked "What about the dog?", the tool call becomes `analyze_images(query="What about `dog_park.jpg`?")`). This is crucial as it allows the `analyze_images` tool to correctly identify and process the specific image from its internal storage or cache.
    - **Case 3 (General Follow-up / No New Files, No Implicit Memory Match):** If no new explicit files and no strong implicit reference from memory, I MUST call `analyze_images` with ONLY the `query` parameter (the original user query). I will NOT use the `file_paths` parameter in this case. The `analyze_images` tool is expected to handle this by perhaps analyzing the *last* uploaded image if relevant, or providing a general analysis.

    **Step 4: Execute and Report.**
    - Call the tool(s) according to the plan.
    - Take the `analysis` text from the final tool output and present it as my final answer, without any modification or added text. Ensure any user preferences noted from `Memory Context` (e.g., conciseness, prioritizing objects) are reflected in the final output style.

    ---

    **Memory Context:** {memory}
    """
    ,
    tools=[upload_images, analyze_images] 
)