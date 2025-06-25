import logging                       
import os
import click
import uvicorn
from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
# Use absolute import for the executor
from Agents.web_search_agent.web_search_executor import ADKWebSearchAgentExecutor

load_dotenv()
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10012)
def main(host: str, port: int):
    # Ensure PERPLEXITY_API_KEY is set for the Perplexity model used via LiteLLM.
    if not os.getenv('PERPLEXITY_API_KEY'):
        logger.error('PERPLEXITY_API_KEY environment variable not set. This agent may not function correctly.')
    
    # Define the capabilities and skills of this Web Search Agent.
    skills = [
        AgentSkill(
            id='web_search',
            name='Web Search',
            description='Searches the web for real-time information and provides up-to-date answers.',
            tags=['search', 'web', 'real-time', 'information', 'research'],
            examples=[
                'What is the latest news about artificial intelligence?',
                'Current stock price of Apple',
                'Recent developments in renewable energy',
                'What happened in the world today?',
            ],
        ),
        AgentSkill(
            id='current_events ',
            name='Current Events',
            description='Provides information about breaking news, trends, and recent developments.',
            tags=['news', 'current events', 'breaking news', 'trends', 'updates'],
            examples=[
                'Latest breaking news today',
                'Current trends in technology',
                'Recent political developments',
                'What are the top stories right now?',
            ],
        ),
        AgentSkill(
            id='research_assistance',
            name='Research Assistance',
            description='Helps with academic, business, and personal research using web sources.',
            tags=['research', 'academic', 'business', 'analysis', 'sources'],
            examples=[
                'Research the history of quantum computing',
                'Find statistics about climate change',
                'What are the latest studies on machine learning?',
                'Business trends in the tech industry',
            ],
        ),
        AgentSkill(
            id='fact_verification',
            name='Fact Verification',
            description='Verifies facts and cross-references information from multiple sources.',
            tags=['fact check', 'verification', 'accuracy', 'sources', 'validation'],
            examples=[
                'Is this news article accurate?',
                'Verify this claim about renewable energy',
                'Cross-check these statistics',
                'What do multiple sources say about this topic?',
            ],
        ),
    ]
    
    agent_executor = ADKWebSearchAgentExecutor()
    agent_card = AgentCard(
        name='ADK Web Search Agent',
        description='I can search the web for real-time information, provide current news and updates, assist with research, and verify facts using multiple sources.',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=skills
    )
    
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )
    
    # Create the A2A application
    a2a_app = A2AStarletteApplication(agent_card, request_handler)
    
    # Build the Starlette app
    app = a2a_app.build()
    
    # Add health endpoint to the Starlette app
    # Note: Starlette apps can have FastAPI-style routes added
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    
    async def health_check(request):
        return JSONResponse({
            "name": agent_card.name,
            "description": agent_card.description,
            "url": agent_card.url,
            "version": agent_card.version,
            "defaultInputModes": agent_card.defaultInputModes,
            "defaultOutputModes": agent_card.defaultOutputModes,
            "capabilities": {
                "streaming": agent_card.capabilities.streaming
            },
            "skills": [
                {
                    "id": skill.id,
                    "name": skill.name,
                    "description": skill.description,
                    "tags": skill.tags,
                    "examples": skill.examples
                }
                for skill in agent_card.skills
            ]
        })
    
    # Add the health route to the existing routes
    health_route = Route("/health", health_check, methods=["GET"])
    app.routes.append(health_route)
    
    logger.info(f"Starting Web Search Agent server on http://{host}:{port}")
    logger.info(f"This agent is identified by 'web_searcher' for delegation.")
    logger.info(f"Health endpoint available at: http://{host}:{port}/health")
    logger.info("🔍 Web Search Agent - Powered by Perplexity AI")
    logger.info("🌐 Capabilities: Real-time Search • Current Events • Research • Fact Verification")
    logger.info("📊 Features: Breaking News • Stock Prices • Academic Research • Trend Analysis")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()