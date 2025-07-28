# main.py

import logging
import os
import click
import uvicorn
import anthropic
from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from starlette.responses import JSONResponse
from starlette.routing import Route

from Agents.Excel_agent.excel_executor import ADKExcelAgentExecutor
from Agents.Excel_agent import excel_adk  # Import the module, not the agent

load_dotenv()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10016)  
def main(host: str, port: int):
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error('FATAL: ANTHROPIC_API_KEY environment variable not set. The agent cannot start.')
        return
    
    try:
        client = anthropic.Anthropic(
            api_key=api_key,
            default_headers={"anthropic-beta": "code-execution-2025-05-22,files-api-2025-04-14"}
        )
        # Set the client on the module, not the agent
        excel_adk.set_anthropic_client(client)
        logger.info("Anthropic client initialized successfully and set for agent tools.")
    except Exception as e:
        logger.error(f"FATAL: Could not initialize Anthropic client: {e}")
        return

    skills = [
        AgentSkill(
            id='tabular_data_expert',
            name='file_analyzer',
            description=(
                'This skill allows you to fully analyze CSV and Excel files. '
                'You can **clean data**, **answer questions** through calculations and queries, '
                '**create charts and graphs**, and provide **summaries and insights**.'
            ),
            tags=[
                'data analysis', 'excel', 'csv', 'data cleaning', 'data query',
                'data calculation', 'data visualization', 'data summary', 'insights'
            ],
            examples=[
                'Clean `sales.csv` by removing duplicates.',
                'What is the total revenue in `report.xlsx`?',
                'Create a bar chart of `regions.csv` showing sales by region.',
                'Summarize the key findings from `feedback.xlsx`.',
                'Compare the performance in `Q1.csv` and `Q2.csv`.',
                'Generate a line graph of `stock_prices.csv` over time.',
                'Calculate the average age from `customers.csv`.'
            ],
        ),
    ]

    agent_executor = ADKExcelAgentExecutor()

    agent_card = AgentCard(
        name='ADK Anthropic Multi-Tool Data Analyst',
        description = "I am a multi-functional file analysis agent utilizing Claude's Code Interpreter for two distinct modes of operation: 1) For Tabular Data (CSV, XLSX), I execute Python code (pandas, matplotlib) to perform deep quantitative analysis, including data cleaning, statistical modeling, and charting. 2) For Text Documents (PDF), I employ text extraction and natural language processing to provide summaries, identify themes, and answer questions about the content. This dual capability allows me to serve as both a data analyst and a research assistant.",
        url=f'http://{host}:{port}/',
        version='2.0.0', 
        defaultInputModes=['text', 'file'],
        defaultOutputModes=['text', 'file'], 
        capabilities=AgentCapabilities(streaming=True),
        skills=skills
    )

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )
    a2a_app = A2AStarletteApplication(agent_card, request_handler)
    app = a2a_app.build()

    async def health_check(request):
        return JSONResponse(agent_card.dict(by_alias=True, exclude_none=True))

    app.routes.append(Route("/health", health_check, methods=["GET"]))

    logger.info(f"Starting Anthropic Multi-Tool Data Analyst server on http://{host}:{port}")
    logger.info(f"Health endpoint available at: http://{host}:{port}/health")
    logger.info("🚀 Agent is ready to receive tasks.")

    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()