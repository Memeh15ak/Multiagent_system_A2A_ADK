# adk_agent.py - FIXED VERSION WITH CLEANER CODE OUTPUT
import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
from anthropic import Anthropic
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    """Generates clean, ready-to-use code with minimal explanation"""
    if not anthropic_client:
        return {"status": "error", "error_message": "Anthropic client not initialized"}

    try:
        # More specific prompt for code generation
        prompt = f"""Generate Python code for: {request}

Requirements:
1. Return clean, working Python code
2. Include the code in ```python blocks
3. Make it ready to copy and run
4. code generated should have best time and space complexity

Request: {request}

Format:
```python
# Brief description
def function_name():
    # implementation
    pass
```

Example usage (if helpful):
# function_name()"""

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

        # Return clean format that's easy to parse
        final_result ={
            "status": "success", 
            "code_generated": True,
            "result": result.strip(),
            "type": "code_generation"
        }
        logger.info(f"Result: {final_result['result']}")
        logger.info(f"Generated code: {final_result}")
        return final_result

    except Exception as e:
        return {"status": "error", "error_message": f"Code generation error: {str(e)}"}

def analyze_and_run_code(request: str) -> dict:
    """Analyzes existing code and provides execution results"""
    if not anthropic_client:
        return {"status": "error", "error_message": "Anthropic client not initialized"}

    try:
        prompt = f"""Analyze and/or execute this code request: {request}

Tasks:
1. If there's code to run, execute it and show output
2. If it's analysis, provide detailed breakdown
3. Use code execution tool when needed
4. Be thorough in your analysis

Request: {request}"""

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
                    
        
        response = {
            "status": "success", 
            "code_generated": False,
            "result": result.strip(),
            "type": "code_analysis"
        }
        logger.info(f"output generated for analyse and run: {response}")
        return response
    except Exception as e:
        return {"status": "error", "error_message": f"Code analysis error: {str(e)}"}

# IMPROVED AGENT WITH BETTER INTENT DETECTION
create_code_execution_agent = Agent(
    name="Smart_Code_Agent",
    model=LiteLlm(model="claude-sonnet-4-20250514"), 
    description="Smart code agent that generates clean code or analyzes existing code",
    instruction="""You are a Smart Code Agent that helps with code generation and analysis.

🎯 DECISION LOGIC - CHOOSE THE RIGHT FUNCTION:

**FOR CODE GENERATION** (User wants NEW code created):
→ Use generate_code_only()

GENERATION INDICATORS:
- "generate code for..."
- "create a function for..."
- "write code to..."
- "make a program for..."
- "code for [something]"
- "function to [do something]"
- "build a [something]"

EXAMPLES:
✅ "generate me a code for multiplication of 2 nos" → generate_code_only()
✅ "create a calculator function" → generate_code_only()
✅ "write code to sort a list" → generate_code_only()
✅ "make a palindrome checker" → generate_code_only()

**FOR CODE ANALYSIS** (User has existing code or wants analysis):
→ Use analyze_and_run_code()

ANALYSIS INDICATORS:
- "analyze this code: [code]"
- "run this program: [code]"
- "debug my function"
- "test this code"
- "explain this code"
- User provides actual code blocks

EXAMPLES:
✅ "analyze this code: def multiply(a,b): return a*b" → analyze_and_run_code()
✅ "run this program and show output" → analyze_and_run_code()

🔥 KEY RULE: 
- If user mentions "generate/create/write/make code FOR [something]" → They want NEW CODE → generate_code_only()
- If user provides existing code or asks to analyze/run something → analyze_and_run_code()

RESPONSE FORMAT:
Always structure your response clearly:
1. State which function you're calling and why
2. Call the appropriate function with the full user query
3. Present the result in a clear, easy-to-read format
4. AND IF A QUERY IS TO GENERATE CODE JUST GENERATE THE CODE AND GIVE BACK THE CODE , DO DONOT ANALYZE THAT 
5. IF THE QUERY IS ABOUT ANALYZING , DO DONT HALLUCINATE AND GENERATE CODE 

Make ONLY ONE function call per request.""",
    tools=[generate_code_only, analyze_and_run_code],
)