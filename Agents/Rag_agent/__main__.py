import logging
import os
import click
import uvicorn
import google.generativeai as genai
from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from starlette.responses import JSONResponse
from starlette.routing import Route

# Import the updated executor
from Agents.Rag_agent.rag_executor import ADKFileRagExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10019)
def main(host: str, port: int):
    """Start the RAG Agent server."""
    
    # Check for required environment variables
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error('FATAL: GEMINI_API_KEY environment variable not set. The agent cannot start.')
        logger.error('Please set your Gemini API key in your .env file or environment.')
        return
    
    # Configure Google GenAI
    try:
        genai.configure(api_key=api_key)
        logger.info("Google GenAI client configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure Google GenAI: {e}")
        return
    
    # Define agent skills
    skills = [
        AgentSkill(
            id='document_analysis',
            name='Comprehensive Document Analysis & Q&A',
            description=(
                "Performs deep analysis on one or more documents (PDF, DOCX, TXT, MD, CSV, etc.) to answer user queries. "
                "Capabilities include generating concise summaries, comparing and contrasting information across multiple files, "
                "extracting specific facts, and providing detailed answers to targeted questions. All responses are strictly "
                "grounded in the content of the provided documents with proper citations."
            ),
            tags=[
                'RAG', 'document analysis', 'Q&A', 'summary', 'comparison',
                'fact-checking', 'text extraction', 'retrieval-augmented generation',
                'multi-file analysis', 'citation-based answers'
            ],
            examples=[
                'Analyze the annual_report.pdf and tell me what was the total revenue in Q4?',
                'Compare the conclusions of study_A.pdf and study_B.pdf, highlighting key differences.',
                'Summarize the attached project proposal and extract all action items with deadlines.',
                'What are the main findings from the research papers in the uploaded files?',
                'Cross-reference the budget.xlsx and project_plan.docx to identify any discrepancies.'
            ],
        ),
        AgentSkill(
            id='multi_file_comparison',
            name='Multi-File Comparison & Analysis',
            description=(
                "Compares and analyzes multiple documents simultaneously, identifying similarities, differences, "
                "contradictions, and complementary information. Excellent for research synthesis, document review, "
                "and cross-referencing multiple sources."
            ),
            tags=[
                'comparison', 'multi-file', 'synthesis', 'cross-reference',
                'research analysis', 'document review'
            ],
            examples=[
                'Compare the methodologies across all three research papers.',
                'Identify any contradictions between the policy documents.',
                'Synthesize findings from all uploaded reports into a single summary.'
            ],
        ),
    ]
    
    # Create agent executor
    try:
        agent_executor = ADKFileRagExecutor()
        logger.info("RAG Agent executor initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Agent executor: {e}")
        return
    
    # Create agent card
    agent_card = AgentCard(
        name='Advanced Multi-File RAG Agent',
        description=(
            "A powerful RAG (Retrieval-Augmented Generation) agent that analyzes one or more files to answer user questions. "
            "It leverages Google's Gemini model to read and comprehend various file types including PDFs, Word documents, "
            "text files, CSVs, and more. Provide your files and a query to receive detailed, citation-backed answers based "
            "solely on the documents' content. Perfect for research, document analysis, and multi-source information synthesis."
        ),
        url=f'http://{host}:{port}/v1/tasks',
        version='2.0.0',
        defaultInputModes=['text', 'file'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=skills
    )
    
    # Create request handler and app
    try:
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor, 
            task_store=InMemoryTaskStore()
        )
        
        a2a_app = A2AStarletteApplication(agent_card, request_handler)
        app = a2a_app.build()
        
        # Add health check endpoint
        async def health_check(request):
            return JSONResponse({
                "status": "healthy",
                "agent": agent_card.name,
                "version": agent_card.version,
                "capabilities": agent_card.capabilities.model_dump() if agent_card.capabilities else {},
                "skills": len(skills),
                "endpoints": {
                    "health": f"http://{host}:{port}/health",
                    "tasks": agent_card.url
                }
            })
        
        health_route = Route("/health", health_check, methods=["GET"])
        app.routes.append(health_route)
        
        logger.info("=" * 60)
        logger.info(f"🚀 Starting {agent_card.name} v{agent_card.version}")
        logger.info("=" * 60)
        logger.info(f"Server: http://{host}:{port}")
        logger.info(f"Health Check: http://{host}:{port}/health")
        logger.info(f"Task Endpoint: {agent_card.url}")
        logger.info(f"Supported Files: PDF, DOCX, TXT, MD, CSV, JSON, XML, and more")
        logger.info(f"Available Skills: {len(skills)}")
        logger.info("=" * 60)
        logger.info("✅ Agent is ready to receive tasks!")
        
        # Start the server
        uvicorn.run(app, host=host, port=port)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return

if __name__ == '__main__':
    main()