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
from Agents.img_to_img_agent.img_executor import ADKImageToImageAgentExecutor

load_dotenv()
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10014)
def main(host: str, port: int):
    # Ensure GEMINI_API_KEY is set for the Gemini model used for image generation.
    if not os.getenv('GEMINI_API_KEY'):
        logger.error('GEMINI_API_KEY environment variable not set. This agent may not function correctly.')
    
    # Define the capabilities and skills of this Image-to-Image Agent.
    skills = [
        AgentSkill(
            id='image_transformation',
            name='Image Transformation',
            description='Transform and modify images based on text descriptions using AI.',
            tags=['image', 'transformation', 'ai', 'modification', 'generation'],
            examples=[
                'Add a sunset background to this photo',
                'Change the style of this image to watercolor',
                'Add a cat to this landscape image',
                'Transform this photo into a cartoon style',
            ],
        ),
        AgentSkill(
            id='style_transfer',
            name='Style Transfer',
            description='Apply artistic styles and filters to images.',
            tags=['style', 'artistic', 'filter', 'art', 'creative'],
            examples=[
                'Make this image look like a Van Gogh painting',
                'Apply oil painting style to this photo',
                'Convert this image to black and white with vintage effect',
                'Add impressionist style to this landscape',
            ],
        ),
        AgentSkill(
            id='object_addition',
            name='Object Addition',
            description='Add objects, people, or elements to existing images.',
            tags=['object', 'addition', 'insertion', 'composition', 'editing'],
            examples=[
                'Add a rainbow to this sky',
                'Place a dog in this park scene',
                'Add flowers to this garden image',
                'Insert a mountain range in the background',
            ],
        ),
        AgentSkill(
            id='scene_modification',
            name='Scene Modification',
            description='Modify scenes, lighting, weather, and environmental elements in images.',
            tags=['scene', 'environment', 'lighting', 'weather', 'atmosphere'],
            examples=[
                'Change this sunny day to a rainy scene',
                'Add snow to this winter landscape',
                'Make this indoor scene brighter',
                'Change the time of day to golden hour',
            ],
        ),
        AgentSkill(
            id='image_enhancement',
            name='Image Enhancement',
            description='Enhance and improve image quality, colors, and details.',
            tags=['enhancement', 'quality', 'color', 'detail', 'improvement'],
            examples=[
                'Enhance the colors in this faded photo',
                'Improve the lighting in this dark image',
                'Make this image more vibrant and sharp',
                'Restore details in this blurry photo',
            ],
        ),
    ]
    
    agent_executor = ADKImageToImageAgentExecutor()
    agent_card = AgentCard(
        name='ADK Image-to-Image Agent',
        description='I can transform and modify images based on your text descriptions. Upload an image and tell me how you want to change it - I can add objects, change styles, modify scenes, enhance quality, and much more!',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=['text', 'image'],
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
    
    # Add health endpoint to the Starlette app
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
    
    logger.info(f"Starting Image-to-Image Agent server on http://{host}:{port}")
    logger.info(f"This agent is identified by 'image_transformer' for delegation.")
    logger.info(f"Health endpoint available at: http://{host}:{port}/health")
    logger.info("🎨 Image-to-Image Agent - Powered by Gemini 2.0 Flash Preview")
    logger.info("🖼️  Capabilities: Image Transformation • Style Transfer • Object Addition • Scene Modification • Image Enhancement")
    logger.info("✨ Features: Artistic Styles • Object Insertion • Lighting Changes • Quality Enhancement • Creative Transformations")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()