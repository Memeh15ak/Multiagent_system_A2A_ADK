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
from Agents.code_executor_agent.code_exe import ADKCodeExecutionAgentExecutor

load_dotenv()
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10011)
def main(host: str, port: int):
    # Check for required API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.error('ANTHROPIC_API_KEY environment variable not set. This agent may not function correctly.')
    
    # Define the capabilities and skills of this Code Execution Agent.
    skills = [
        AgentSkill(
            id='execute_code',
            name='Execute Code',
            description='Executes Python code, performs calculations, and creates visualizations. When u r giving a code u should run it and provide what willbe its output as well',
            tags=['code', 'python', 'execution', 'calculation', 'visualization'],
            examples=[
                'Calculate the derivative of x^2',
                'Create a plot of sine wave',
                'Run this Python code: print("Hello World")',
                'Solve this math problem: 2x + 5 = 15',
            ],
        ),
        AgentSkill(
            id='analyze_files',
            name='Analyze Files',
            description='Analyzes various file types including Python files, CSV, Excel, PDFs, images, and text files.If givinga code file or any task u should run it and provide its output as a response as well',
            tags=['file', 'analysis', 'csv', 'excel', 'python', 'pdf', 'data'],
            examples=[
                'Analyze this CSV file and show summary statistics',
                'Run this Python file: script.py',
                'Analyze data.xlsx and create visualizations',
                'Debug this Python code file',
            ],
        ),
        AgentSkill(
            id='data_science',
            name='Data Science & ML',
            description='Performs data science tasks, machine learning, and statistical analysis.',
            tags=['data science', 'machine learning', 'statistics', 'ml', 'analysis'],
            examples=[
                'Train a linear regression model on this data',
                'Perform statistical analysis on sales data',
                'Create a machine learning pipeline',
                'Generate correlation matrix visualization',
            ],
        ),
    ]
    
    agent_executor = ADKCodeExecutionAgentExecutor()
    agent_card = AgentCard(
        name='ADK Code Execution Agent',
        description='I can execute code, analyze files, perform calculations, create visualizations, and handle data science tasks.',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=['text', 'file'],
        defaultOutputModes=['text', 'image'],
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
    
    # Add health endpoint that returns the full AgentCard structure
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    
    async def health_check(request):
        # Return the full AgentCard structure that the orchestrator expects
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
    
    async def agent_info(request):
        # Alternative endpoint that returns the same information
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
    
    # Add the health and agent info routes
    health_route = Route("/health", health_check, methods=["GET"])
    agent_route = Route("/agent", agent_info, methods=["GET"])
    app.routes.extend([health_route, agent_route])
    
    logger.info(f"Starting Code Execution Agent server on http://{host}:{port}")
    logger.info(f"This agent is identified by 'code_executor' for delegation.")
    logger.info(f"Health endpoint available at: http://{host}:{port}/health")
    logger.info(f"Agent info endpoint available at: http://{host}:{port}/agent")
    logger.info("🚀 Code Execution Agent - Powered by Anthropic Claude")
    logger.info("📊 Capabilities: Code Execution • File Analysis • Data Viz • ML • Debug")
    logger.info("📁 Supported: .py .csv .xlsx .pdf .txt .json .xml .html .js .css .md")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()