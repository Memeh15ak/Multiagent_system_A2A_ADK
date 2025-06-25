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
from Agents.report_gen_agent.report_executor import ADKReportAgentExecutor

load_dotenv()
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10015)
def main(host: str, port: int):
    # Ensure PERPLEXITY_API_KEY is set for the Perplexity model used via LiteLLM.
    if not os.getenv('PERPLEXITY_API_KEY'):
        logger.error('PERPLEXITY_API_KEY environment variable not set. This agent may not function correctly.')
    
    # Define the capabilities and skills of this Web Search Agent.
    skills = [
        AgentSkill(
            id='report_generation',
            name='Report Generation',
            description='Generates detailed reports based on web searches and current events.',
            tags=['report', 'generates'],
            examples=[
                'Generate a report on the Latest AI Trends',
                'Create a summary report on the current state of renewable energy',
                'Generate a market analysis report for the tech industry',
            ],
        ),
        AgentSkill(
            id='content_generation',
            name='Content Generation',
            description='Creates articles, summaries, searches and other content based on the given topic ',
            tags=['content','generation', 'articles','summaries', 'searches'],
            examples=[
                'Write an article about the future of space exploration',
                'Give me some lines on the impact of AI on society',
                'Summarize the latest reasearch on climate change',
            ],
        ),
        
    ]
    
    agent_executor = ADKReportAgentExecutor()
    agent_card = AgentCard(
        name='ADK Report and Content Generation Agent',
        description='An advanced agent for generating reports and content based on web searches , current events, creativity , proper formatting and more.',
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
    
    logger.info(f"Starting Report and content gen  http://{host}:{port}")
    logger.info(f"This agent is identified by 'report gen' for delegation.")
    logger.info(f"Health endpoint available at: http://{host}:{port}/health")
    logger.info("🔍 report gen  Agent - Powered by Perplexity AI")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()