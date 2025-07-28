import os
import time
import logging
import docx
import google.generativeai as genai

from google.adk.agents import Agent
from a2a.server.tasks.task_updater import Message  # Add this import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_FOR_ADK_AGENT = "gemini-1.5-flash-latest" 
MODEL_FOR_ANALYSIS = "gemini-2.0-flash"

def extract_text_from_docx(path: str) -> str | None:
    """Opens and reads a .docx file, returning its text content."""
    try:
        document = docx.Document(path)
        return '\n'.join([para.text for para in document.paragraphs])
    except Exception as e:
        logger.warning(f"Could not read docx file {os.path.basename(path)}: {e}")
        return None

def analyze_multiple_files(file_paths: list[str], query: str) -> dict:
    """
    Analyzes content from a list of files to answer a user's query.
    Handles .docx files by extracting text, and uploads other supported file types.
    """
    logger.info(f"--- Tool Activated: analyze_multiple_files ---")
    logger.info(f"  Query: {query}")
    logger.info(f"  Files: {file_paths}")

    prompt_parts = []
    files_to_delete = []

    for path in file_paths:
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}. Skipping.")
            continue

        file_basename = os.path.basename(path)
        
        if path.lower().endswith('.docx'):
            logger.info(f"Extracting text from DOCX: {file_basename}...")
            text_content = extract_text_from_docx(path)
            if text_content:
                formatted_text = f"--- Content from {file_basename} ---\n{text_content}\n--- End of {file_basename} ---"
                prompt_parts.append(formatted_text)
        else:
            logger.info(f"Uploading: {file_basename}...")
            try:
                uploaded_file = genai.upload_file(path=path)
                prompt_parts.append(uploaded_file)
                files_to_delete.append(uploaded_file)
            except Exception as e:
                logger.warning(f"Could not upload '{file_basename}'. It might be an unsupported file type. Error: {e}")

    if not prompt_parts:
        return(f"Error in prompt_parts: No valid files found or all files were unsupported. Please check the file paths and types.")

    for file in files_to_delete:
        while file.state.name == "PROCESSING":
            logger.info(f"Waiting for {file.display_name}...")
            time.sleep(5)
            file = genai.get_file(name=file.name)
        if file.state.name == "FAILED":
            for f in files_to_delete: genai.delete_file(f.name)
            return {f"File processing failed for {file.display_name}"}

    logger.info("All content ready. Asking Gemini for analysis...")
    try:
        system_instruction = """You are a meticulous analyst. Your task is to answer the user's query based ONLY on the provided documents.

        **CRITICAL RULES:**
        1.  Treat each document (or text snippet) as a completely separate and distinct source.
        2.  **DO NOT** mix, merge, or blend facts between different documents.
        3.  If information comes from a specific file, you must cite that file in your answer. For example, "According to a.txt...".
        4.  If the answer to a part of the query is not in any document, you must state that explicitly.
        """

        final_prompt = [system_instruction, "--- USER QUERY ---", query, "--- PROVIDED DOCUMENTS ---"] + prompt_parts
        
        model = genai.GenerativeModel(MODEL_FOR_ANALYSIS)
        response = model.generate_content(final_prompt)
        
        return {"status": "success", "analysis": response.text.strip()}
    except Exception as e:
        logger.error(f"Error during Gemini content generation: {e}", exc_info=True)
    finally:
        if files_to_delete:
            logger.info("Cleaning up uploaded files...")
            for file in files_to_delete:
                genai.delete_file(name=file.name)

memory = [
    "The user previously asked about 'project_update.docx' for its key points.",
    "The last file analyzed was 'Q4_report.pdf'. It contained sales figures and challenges.",
    "User prefers summaries to be in bullet points.",
    "User tends to ask follow-up questions about comparative analysis.",
    "The 2023 budget file 'budget_2023.xlsx' showed an allocated marketing budget of $50,000.",
    "I like reading sci-fi novels." 
]

# FIXED: Remove the gemini/ prefix from the model name
create_file_rag_agent = Agent(
    name="file_rag_gemini_router",
    model=MODEL_FOR_ADK_AGENT,
    description="Agent that analyzes one or more files by calling the appropriate tool.",
    instruction=(
        f"""You are an expert file analysis assistant. Your purpose is to provide accurate answers to user questions by analyzing the content of local files using your tools.

        **Core Principle:** You must base your answers **exclusively** on the output provided by the `analyze_multiple_files` tool. If the tool's analysis does not contain the answer, you must state that the information is not available in the provided documents. Do not use your general knowledge for file-based questions.

        **Tool Usage Rules:**
        1.  When a user's query contains one or more file names (e.g., `report.pdf`, `data.csv`, `notes.docx`), or **implicitly refers to a previously known file from `Memory Context`**, you MUST call the `analyze_multiple_files` tool.
        2.  You MUST extract all explicitly mentioned file paths from the user's query, AND **if the query implicitly refers to a file found in `Memory Context` (e.g., "the last report", "that spreadsheet", "the budget from 2023"), include its filename(s) in the list of `file_paths` to pass to the tool.** All files should be passed as a Python list of strings to the `file_paths` argument.
        3.  You MUST create a clear and specific `query` argument for the tool that accurately reflects the user's intent. **Leverage `Memory Context` to make the `query` more precise or to include relevant background facts that the tool might need for context.** (e.g., if memory states "The 2023 budget... showed an allocated marketing budget of $50,000", and the user asks "How does this year's budget compare?", the tool query could be "Compare this year's budget to 2023, where 2023 marketing budget was $50,000.")
        4.  When the user's query begins with "Based on the loaded session files...", this is your hint that a pre-loaded context exists. You should still look for *additional* file names in the rest of the query and pass **all** relevant files to the tool.
        5.  If the tool returns an error message (e.g., "File not found"), you must relay that error clearly and politely to the user.

        ### EXAMPLES OF CORRECT TOOL USAGE ###

        **Example 1: Single File Summary**
        User Query: "Can you give me the key points from 'project_update.docx'?"
        Tool Call: `analyze_multiple_files(file_paths=['project_update.docx'], query='Extract the key points from this document.')`

        **Example 2: Multi-File Comparison**
        User Query: "Please compare the methodology in 'study_A.pdf' with the one in 'study_B.pdf'."
        Tool Call: `analyze_multiple_files(file_paths=['study_A.pdf', 'study_B.pdf'], query='Compare the methodology sections of these two documents, highlighting similarities and differences.')`

        **Example 3: Query on Pre-loaded Session Files**
        User Query: "Based on the loaded session files ('plan.docx', 'budget.csv'), does the plan fit within the budget?"
        Tool Call: `analyze_multiple_files(file_paths=['plan.docx', 'budget.csv'], query='Analyze the provided plan and budget files to determine if the plan is feasible within the given budget.')`
        
        **Example 4: Mixed Query (Session + New File)**
        User Query: "Based on the loaded session files ('main_report.pdf'), how do its conclusions differ from the new data in 'appendix.csv'?"
        Tool Call: `analyze_multiple_files(file_paths=['main_report.pdf', 'appendix.csv'], query='Compare the conclusions of the main report with the new data in the appendix file.')`
        
        ---

        **General Memory Handling Instructions:**
        - Always review the `Memory Context` provided below.
        - **Only use information from `Memory Context` if it is directly relevant to your specific role (file analysis) and the current user query.**
          *   For example, if `Memory Context` contains a fact like "The last file analyzed was 'Q4_report.pdf'", and the user asks "What were the challenges?", you should infer they mean 'Q4_report.pdf' and include it in your `file_paths` to the tool, and potentially enrich the tool's `query` with "regarding sales figures and challenges from Q4".
          *   If `Memory Context` states "User prefers summaries to be in bullet points", incorporate this preference into your generated tool query (e.g., `query='Summarize this document in bullet points.'`).
          *   If `Memory Context` states "I like reading sci-fi novels" and the query is "Summarize `document.pdf`", this memory is not relevant to file analysis, so you MUST ignore it for this query.
        - If information in `Memory Context` is not relevant to your current task or query, you MUST ignore it and proceed with your core responsibilities. Do not try to answer questions outside your domain based on this memory.

        **Memory Context:** {memory}
        """
    ),
    tools=[analyze_multiple_files], 
)