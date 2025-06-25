# adk_agent.py - SUPER SIMPLE VERSION THAT ACTUALLY WORKS
import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

try:
    anthropic_client = Anthropic(
        default_headers={
            "anthropic-beta": "code-execution-2025-05-22,files-api-2025-04-14"
        }
    )
except Exception as e:
    print(f"Error initializing Anthropic client: {e}")
    anthropic_client = None

def generate_code_only(request: str) -> dict:
    """ONLY generates code - NO analysis, NO explanations"""
    if not anthropic_client:
        return {"status": "error", "error_message": "Anthropic client not initialized"}

    try:
        prompt = f"""Generate code for: {request}

Return ONLY the code in ```python blocks. No explanations. No analysis. Just code.

Examples:
- "fibonacci using recursion" → return recursive fibonacci function
- "calculator" → return calculator code
- "sort array" → return sorting code

Request: {request}"""

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        result = ""
        if response.content:
            for block in response.content:
                if block.type == "text":
                    result += block.text

        return {"status": "success", "result": result.strip()}

    except Exception as e:
        return {"status": "error", "error_message": f"Error: {str(e)}"}

def analyze_and_run_code(request: str) -> dict:
    """Analyzes code, runs it, provides output"""
    if not anthropic_client:
        return {"status": "error", "error_message": "Anthropic client not initialized"}

    try:
        prompt = f"""Analyze/run this code request: {request}

If there's code to run, execute it and provide output.
If it's analysis, provide detailed analysis.
Use code execution tool when needed.
"""

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "code_execution_20250522", "name": "code_execution"}]
        )

        result = ""
        if response.content:
            for block in response.content:
                if block.type == "text":
                    result += block.text

        return {"status": "success", "result": result.strip()}

    except Exception as e:
        return {"status": "error", "error_message": f"Error: {str(e)}"}

# CRYSTAL CLEAR AGENT INSTRUCTIONS
create_code_execution_agent = Agent(
    name="Simple_Code_Agent",
    model=LiteLlm(model="claude-sonnet-4-20250514"), 
    description="Simple code agent that generates or analyzes code",
    instruction="""You are a Simple Code Agent. Your job is to look at the user's request and call the RIGHT function.

🔥 DECISION RULES (SUPER SIMPLE):

1. If user wants CODE GENERATION (generate, create, write, build, make, code for):
   → Call generate_code_only()

2. If user wants ANALYSIS/EXECUTION (analyze, run, execute, debug, output, test):
   → Call analyze_and_run_code()

GENERATION KEYWORDS: generate, create, write, build, make, code, function, program
ANALYSIS KEYWORDS: analyze, run, execute, debug, output, test, check

EXAMPLES:
✅ "generate fibonacci code" → generate_code_only("generate fibonacci code")
✅ "create calculator" → generate_code_only("create calculator")  
✅ "analyze this code" → analyze_and_run_code("analyze this code")
✅ "run fibonacci program" → analyze_and_run_code("run fibonacci program")

IMPORTANT:
- Don't overthink it
- Just match keywords and call the right function
- Pass the FULL user query to the function
- Let the function handle the rest""",
    tools=[generate_code_only, analyze_and_run_code],
)