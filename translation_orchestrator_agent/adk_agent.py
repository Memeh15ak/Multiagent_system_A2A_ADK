# simplified_adk_agent.py - ADK agent with complete agent card awareness
import os
import logging
from dotenv import load_dotenv
from google.adk.agents import LlmAgent

# Import the simplified discovery system
from translation_orchestrator_agent.dynamic_agent_discovery import SimpleAgentCardDiscovery
# Import the tools
from translation_orchestrator_agent.a2a_translation_tools import (
    web_search_function,
    code_execution_function,
    image_modification_function,
    audio_conversational_function,
    report_and_content_generation_function,
    excel_file_analysis_function,
    rag_agent_function,
    image_analyzer_function,
    image_generation_function,
    video_function
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Global discovery instance
card_discovery = SimpleAgentCardDiscovery()

def safe_get_skills_count(full_card):
    """Safely get skills count from card data."""
    if not full_card:
        return 0
    
    skills = full_card.get('skills', [])
    
    if isinstance(skills, int):
        return skills
    elif isinstance(skills, (list, tuple)):
        return len(skills)
    else:
        return 0

async def create_smart_routing_agent() -> LlmAgent:
    """Create ADK agent with complete agent card awareness."""
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    os.environ["GEMINI_API_KEY"] = gemini_api_key

    # Extract complete agent cards
    logger.info("Extracting complete agent cards...")
    await card_discovery.extract_all_agent_cards()
    
    # Log discovery results
    available_count = len([c for c in card_discovery.agent_cards.values() if c.available])
    total_count = len(card_discovery.agent_cards)
    logger.info(f"Agent card extraction complete. Available: {available_count}/{total_count}")
    
    # Print summary for debugging
    print(card_discovery.get_available_agents_summary())

    # Create the ADK agent with dynamic routing capability
    routing_instruction = """You are an INTELLIGENT AGENT ROUTER that MUST analyze user queries and route them to specialized agents.

🎯 CRITICAL ROUTING RULES:
1. You MUST call a function for EVERY user query - NEVER answer directly
2. Analyze the query type and route to the appropriate specialized agent
3. For information queries (news, current events, research) → call web_search_function
4. For code execution or programming tasks → call code_execution_function, Do not use your own knowledge, if the query is about code generation dpo not generate the code in reponse rather send the query to code_execution_function
5. Let the specialized agent handle the actual task
6.If there are .pdf file route to rag agent and if there is.jpg or any image mime type decide whether its the query to generate an image or analyze the image and route to image_generation_function or image_analyzer_function respectively
7.If there is image modification request like changes in image caoll image_modification_function and if a query is about question answering or summarization of image file route to image_analyzer_function

⚠️ DO NOT:
- Answer queries directly without routing
- Generate code/content and then pass it to functions - pass the original query
- Even if you know the answer - always route to specialized agents
- Say "I don't know" or "hasn't happened yet" - route to web_search_function for current info
- Provide your own knowledge - always route to specialized agents

✅ SUCCESS PATTERN:
User: "Who won IPL 2025?"
You: [Call web_search_function with query "IPL 2025 winner results"]
Agent Response: [Current information from web search]
You: "I routed your query to the Web Search Agent. Here's the complete response: [FULL RESPONSE]"

User: "Write a Python function to add two numbers"
You: [Call code_execution_function with query "Write a Python function to add two numbers"]
Agent Response: [Generated Python code]
You: "I routed your query to the Code Execution Agent. Here's the complete response: [FULL RESPONSE]"

🎯 REMEMBER: Your job is ROUTING, not answering. Always call the appropriate function first!"""

     


    return LlmAgent(
        model='gemini-1.5-flash',
        name='intelligent_agent_router',
        description='Routes queries to specialized agents based on complete capability analysis',
        instruction=routing_instruction,
        tools=[
            web_search_function,
            code_execution_function,
            image_modification_function,
            audio_conversational_function,
            report_and_content_generation_function,
            excel_file_analysis_function,
            rag_agent_function,
            image_analyzer_function,
            image_generation_function,
            video_function

        ]
        
    )

async def process_user_query(agent: LlmAgent, user_query: str) -> str:
    """Process user query with complete agent card context."""
    
    # Generate prompt with complete agent cards
    enhanced_prompt = card_discovery.generate_adk_prompt_with_agent_cards(user_query)
    
    # Log the prompt for debugging (optional)
    logger.debug(f"Enhanced prompt length: {len(enhanced_prompt)} characters")
    
    try:
        # Send the enhanced prompt to the ADK agent
        response = await agent.run_async(enhanced_prompt)
        logger.info(f"Agent response: {response}")  # Log first 200 characters for brevity
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"Error processing your query: {str(e)}"

async def refresh_agent_cards():
    """Refresh agent cards - useful for long-running sessions."""
    global card_discovery
    logger.info("Refreshing agent cards...")
    await card_discovery.extract_all_agent_cards()
    logger.info("Agent cards refreshed")
    print(card_discovery.get_available_agents_summary())

def get_agent_status():
    """Get current status of all agents with their cards."""
    status = {}
    for agent_id, card_info in card_discovery.agent_cards.items():
        status[agent_id] = {
            'available': card_info.available,
            'url': card_info.url,
            'has_card': card_info.full_card is not None,
            'card_summary': {
                'name': card_info.full_card.get('name') if card_info.full_card else None,
                'description': card_info.full_card.get('description') if card_info.full_card else None,
                'skills_count': safe_get_skills_count(card_info.full_card)
            }
        }
    return status

# Example usage
async def main():
    """Example of how to use the simplified ADK agent."""
    
    # Create the agent
    agent = await create_smart_routing_agent()
    
    # Example queries
    test_queries = [
        "Search for recent developments in quantum computing",
        "Write Python code to sort a list of dictionaries by a key",
        "Create a comprehensive report on renewable energy trends",
        "Process this audio file and extract key information",
        "Edit this image to make it more vibrant"
        "generate a code for addition of two numbers"
        "Create a video summarizing the latest tech news",
        "Analyze this Excel file and summarize the data",
        "What are the latest advancements in AI?",
        "Generate an image of a futuristic cityscape",
        "What is the current weather in New York City?"
    ]
    
    print("\n" + "="*80)
    print("TESTING SMART ROUTING WITH COMPLETE AGENT CARDS")
    print("="*80)
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        
        try:
            response = await process_user_query(agent, query)
            print(f"✅ Response: {response[:200]}..." if len(response) > 200 else f"✅ Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())