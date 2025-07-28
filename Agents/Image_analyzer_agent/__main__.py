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

from Agents.Image_analyzer_agent.image_anal_executor import ADKImageAgentExecutor 

load_dotenv()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10017) 
def main(host: str, port: int):
    api_key = os.getenv('GEMINI_API_KEY') 
    if not api_key:
        logger.error('FATAL: GEMINI_API_KEY environment variable not set. The agent cannot start.')
        return
    genai.configure(api_key=api_key)
    logger.info("Google GenAI client configured successfully.")


    skills = [
        AgentSkill(
            id='Image_analyzer',
            name='Comprehensive Image Analysis & Visual Q&A',
            description='Performs detailed visual analysis of images, including identifying objects, scenes, and actions. It can answer specific questions about image content, describe characteristics, and even compare multiple images to highlight similarities or differences.',
            tags=[
                'image analysis', 'description', 'object detection',
                'scene understanding', 'visual Q&A', 'comparison',
                'multi-image analysis', 'information extraction'
            ],
            examples=[
                'What is the main subject of this picture?',
                'Describe everything you see in the attached image.',
                'What are the key differences between these two photos?',
                'How many people are in the first image, and what are they doing?',
                'Can you identify the objects present in this photo?',
                'Compare the provided images and highlight any changes.',
                'What is the overall scene depicted in the image?',
            ],
        ),
    ]

    agent_executor = ADKImageAgentExecutor() 
    agent_card = AgentCard(
    name='ADK Gemini Advanced Image Agent',
    description = "A specialized agent leveraging the advanced visual understanding capabilities of Google's Gemini model. My core function is to execute detailed image analysis based on your query. Capabilities include: detailed image description, object and scene recognition, answering specific questions about image content, and multi-image comparison. Provide the image file(s) and your analysis objective for a comprehensive report.",
    url=f'http://{host}:{port}/',
    version='1.1.0',
    defaultInputModes=['text', 'file'],
    defaultOutputModes=['text'],
    capabilities=AgentCapabilities(streaming=True),
    skills=skills


    )

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )
    a2a_app = A2AStarletteApplication(agent_card, request_handler)
    app = a2a_app.build()

    async def health_check(request):
        return JSONResponse({
            "name": agent_card.name,
            "description": agent_card.description,
            "url": str(agent_card.url),
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
    health_route = Route("/health", health_check, methods=["GET"])
    app.routes.append(health_route)


    logger.info(f"Starting Gemini Image Agent server on http://{host}:{port}")
    logger.info(f"Health endpoint available at: http://{host}:{port}/health")
    logger.info("🚀 Gemini Image Agent is ready to receive tasks.")

    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()