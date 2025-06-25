import logging
import os
import click
import uvicorn
from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from Agents.audio_agent.audio_executor import ADKAudioAgentExecutor

load_dotenv()
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10013)
def main(host: str, port: int):
    # Ensure GEMINI_API_KEY is set for the Gemini model used for audio processing.
    if not os.getenv('GEMINI_API_KEY'):
        logger.error('GEMINI_API_KEY environment variable not set. This agent may not function correctly.')
    
    # Define the capabilities and skills of this Audio Processing Agent.
    skills = [
        AgentSkill(
            id='audio_transcription',
            name='Audio Transcription',
            description='Transcribe spoken words from audio files into text with high accuracy.',
            tags=['audio', 'transcription', 'speech-to-text', 'voice', 'recognition'],
            examples=[
                'Transcribe this voice memo',
                'Convert this audio recording to text',
                'What did the person say in this audio file?',
                'Extract the spoken content from this audio',
            ],
        ),
        AgentSkill(
            id='audio_analysis',
            name='Audio Content Analysis',
            description='Analyze and understand the content, sentiment, and context of audio recordings.',
            tags=['analysis', 'content', 'sentiment', 'understanding', 'context'],
            examples=[
                'Analyze the sentiment of this voice message',
                'What is the main topic discussed in this audio?',
                'Summarize the key points from this audio recording',
                'Identify the speakers and their emotions in this audio',
            ],
        ),
        AgentSkill(
            id='audio_response',
            name='Audio Response Generation',
            description='Generate natural-sounding audio responses based on input audio and queries.',
            tags=['response', 'generation', 'text-to-speech', 'voice', 'audio-output'],
            examples=[
                'Respond to this voice message with audio',
                'Answer this question in audio format',
                'Create an audio summary of this conversation',
                'Generate a voice response to this inquiry',
            ],
        ),
        AgentSkill(
            id='conversation',
            name='Audio Conversation',
            description='Engage in natural voice conversations, maintaining context and providing relevant responses.',
            tags=['conversation', 'dialogue', 'voice-chat', 'interactive', 'contextual'],
            examples=[
                'Have a voice conversation about this topic',
                'Continue our audio discussion',
                'Respond naturally to my voice questions',
                'Engage in voice-based dialogue',
            ],
        ),
        AgentSkill(
            id='audio_processing',
            name='Audio File Processing',
            description='Process various audio formats and handle audio quality optimization.',
            tags=['processing', 'formats', 'quality', 'optimization', 'enhancement'],
            examples=[
                'Process this MP3 file',
                'Handle this WAV recording',
                'Improve the quality of this audio',
                'Convert between audio formats',
            ],
        ),
    ]
    
    agent_executor = ADKAudioAgentExecutor()
    agent_card = AgentCard(
        name='ADK Audio Processing Agent',
        description='I am an advanced audio processing assistant that specializes in handling voice interactions. Upload audio files and I will transcribe, analyze, and respond with natural-sounding audio responses. I can process various audio formats (MP3, WAV, M4A, FLAC, AAC, OGG) and engage in meaningful voice conversations.',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=['audio', 'text'],  # Primary focus on audio input
        defaultOutputModes=['audio'],  # Audio-only output
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
    
    # Log startup information
    logger.info("=" * 60)
    logger.info(f"🎙️  STARTING AUDIO PROCESSING AGENT")
    logger.info("=" * 60)
    logger.info(f"🌐 Server: http://{host}:{port}")
    logger.info(f"🔧 Agent ID: 'audio_processor' (for delegation)")
    logger.info(f"🩺 Health: http://{host}:{port}/health")
    logger.info("=" * 60)
    logger.info("🎧 AUDIO CAPABILITIES:")
    logger.info("   • Speech-to-Text Transcription")
    logger.info("   • Audio Content Analysis") 
    logger.info("   • Natural Voice Response Generation")
    logger.info("   • Multi-format Audio Processing")
    logger.info("🤖 MODEL: Gemini 2.0 Flash Live (Audio Optimized)")
    logger.info("⚡ OUTPUT MODE: Audio-First Responses")
    logger.info("🔄 STREAMING: Enabled")
    logger.info("=" * 60)
    
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()