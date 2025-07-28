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

from Agents.Image_generation_agent.image_gen_executor import ADKImageGeneratorExecutor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10018)  
def main(host: str, port: int):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error('FATAL: GEMINI_API_KEY environment variable not set. The agent cannot start.')
        return
    
    try:
        genai.configure(api_key=api_key)
        logger.info("Google Gemini client configured successfully.")
        
        models = list(genai.list_models())
        logger.info(f"Available models: {len(models)}")
        
    except Exception as e:
        logger.error(f"FATAL: Could not configure Gemini client: {e}")
        return

    skills = [
        AgentSkill(
            id='generate_image',
            name='AI Image Generation',
            description='Creates high-quality images from detailed text descriptions using Google Gemini 2.0 Flash with native image generation capabilities.',
            tags=['image', 'generation', 'art', 'ai', 'creative', 'visual', 'design'],
            examples=[
                'Create a photorealistic portrait of a cyberpunk warrior in neon-lit Tokyo streets',
                'Generate a serene watercolor landscape of mountain peaks reflected in a crystal lake',
                'Design a modern minimalist logo for a sustainable energy company',
                'Draw a whimsical illustration of animals having a tea party in an enchanted forest',
                'Create concept art for a futuristic space station orbiting Mars'
            ],
        ),
    ]

    agent_executor = ADKImageGeneratorExecutor()

    agent_card = AgentCard(
        name='Gemini AI Image Creator',
        description='Advanced AI image generation powered by Google Gemini 2.0 Flash. I transform your creative ideas into stunning visual art with exceptional detail and quality. Describe what you envision, and I\'ll bring it to life.',
        url=f'http://{host}:{port}/',
        version='2.0.0',
        defaultInputModes=['text'],  
        defaultOutputModes=['text', 'image'], 
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
            **agent_card.dict(by_alias=True, exclude_none=True),
            "status": "healthy",
            "timestamp": __import__('datetime').datetime.now().isoformat()
        })

    app.routes.append(Route("/health", health_check, methods=["GET"]))

    logger.info(f" Starting Gemini AI Image Creator on http://{host}:{port}")
    logger.info(f" Health endpoint: http://{host}:{port}/health")
    logger.info("Ready to generate amazing images from your prompts!")

    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()