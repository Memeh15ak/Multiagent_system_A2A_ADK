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
from pydantic import ValidationError

from Agents.Video_agent.video_executor import ADKVideoAgentExecutor 

load_dotenv() 

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10021) 
def main(host: str, port: int):
    api_key = os.getenv('GEMINI_API_KEY') 
    if not api_key:
        logger.error('FATAL: GEMINI_API_KEY environment variable not set. The agent cannot start.')
        return
    genai.configure(api_key=api_key)
    logger.info("Google GenAI client configured successfully.")

    skills = [
        AgentSkill(
            id='comprehensive_video_analysis',
            name='Comprehensive Video Analysis & Visual Q&A',
            description='Performs detailed visual analysis of videos, identifying objects, scenes, actions, and temporal events. It can answer specific questions about video content, summarize events, describe characteristics, and compare multiple videos.',
            tags=[
                'video analysis', 'description', 'object detection',
                'scene understanding', 'visual Q&A', 'comparison',
                'multi-video analysis', 'event recognition', 'temporal analysis',
                'summarization' 
            ],
            examples=[
                'Summarize the key events in this video.',
                'What is happening at [timestamp] in the attached video?',
                'Describe the main subject and actions in this video.',
                'Compare these two videos and tell me their similarities and differences.',
                'How many people appear in the first video, and what are they wearing?',
                'Identify all significant objects and activities throughout the video.',
                'What is the overall theme or purpose of this recording?',
                'Analyze the provided video clips for unusual activities.'
            ],
        ),
    ]

    agent_executor = ADKVideoAgentExecutor()

    try:
        agent_card = AgentCard(
            name='ADK Gemini Advanced Video Agent',
            description = "A specialized agent leveraging the advanced video understanding capabilities of Google's Gemini model. My core function is to execute detailed video analysis based on your query. Capabilities include: detailed video description, object and scene recognition, answering specific questions about video content, summarizing events, and multi-video comparison. Provide the video file(s) and your analysis objective for a comprehensive report.",
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=['text', 'file'],
            defaultOutputModes=['text'],
            capabilities=AgentCapabilities(streaming=True),
            skills=skills
        )
    except ValidationError as e:
        logger.error(f"Pydantic validation error when creating AgentCard:")
        for error in e.errors():
            logger.error(f"  Field: {error.get('loc')}, Message: {error.get('msg')}, Type: {error.get('type')}")
        logger.error(f"Full validation error details (raw): {e}")
        return 

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )
    a2a_app = A2AStarletteApplication(agent_card, request_handler)
    app = a2a_app.build()

    async def health_check(request):
        agent_url_str = str(agent_card.url) if agent_card.url else None
        
        return JSONResponse({
            "name": agent_card.name,
            "description": agent_card.description,
            "url": agent_url_str,
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


    logger.info(f"Starting Gemini Video Agent server on http://{host}:{port}")
    logger.info(f"Health endpoint available at: http://{host}:{port}/health")
    logger.info("🚀 Gemini Video Agent is ready to receive tasks.")

    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()