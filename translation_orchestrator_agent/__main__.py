# Enhanced main.py with dynamic agent discovery
import asyncio
import functools
import logging
import os
import sys

import click
import uvicorn

from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from translation_orchestrator_agent.adk_agent_executor import ADKOrchestratorAgentExecutor
from translation_orchestrator_agent.adk_agent import get_agent_status

load_dotenv()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10020)
@click.option('--discover-timeout', 'discover_timeout', default=10, help='Timeout for agent discovery in seconds')
def main(host: str, port: int, discover_timeout: int):
    # Ensure ANTHROPIC_API_KEY is set for the orchestrator's model.
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.error('ANTHROPIC_API_KEY environment variable not set for orchestrator. This agent may not function correctly.')
        sys.exit(1)

    logger.info("Starting ADK Translation Orchestrator Agent with Dynamic Discovery...")
    
    # Create agent executor (this will trigger discovery)
    agent_executor = ADKOrchestratorAgentExecutor()
    
    # Get and display discovered agents
    agent_status = get_agent_status()
    
    logger.info("=== AGENT DISCOVERY RESULTS ===")
    available_count = 0
    for agent_id, status in agent_status.items():
        status_icon = "✅" if status['available'] else "❌"
        
        # Extract name and skills_count from the nested structure
        name = status.get('card_summary', {}).get('name') or agent_id
        skills_count = status.get('card_summary', {}).get('skills_count', 0)
        
        logger.info(f"{status_icon} {name} ({agent_id})")
        logger.info(f"   URL: {status['url']}")
        logger.info(f"   Skills: {skills_count}")
        if status['available']:
            available_count += 1
    
    logger.info(f"=== SUMMARY: {available_count}/{len(agent_status)} agents available ===")
    
    if available_count == 0:
        logger.warning("⚠️  No agents are currently available! The orchestrator will have limited functionality.")
        logger.info("💡 Make sure your sub-agents are running on their expected ports:")
        for agent_id, status in agent_status.items():
            logger.info(f"   • {agent_id}: {status['url']}")
    
    # Create dynamic agent card based on discovered agents
    dynamic_skills = []
    
    # Add orchestration skill
    dynamic_skills.append(AgentSkill(
        id='orchestration',
        name='Task Orchestration',
        description='Coordinates between multiple specialized agents to complete complex tasks',
        tags=['orchestration', 'coordination', 'multi-agent', 'workflow'],
        examples=[
            'Research a topic and create a report with visualizations',
            'Analyze data and generate insights with charts',
            'Process multimedia content and provide comprehensive analysis',
        ]
    ))
    
    # Add dynamic discovery skill
    dynamic_skills.append(AgentSkill(
        id='dynamic_routing',
        name='Dynamic Agent Routing',
        description='Automatically discovers available agents and routes tasks to the most appropriate ones',
        tags=['routing', 'discovery', 'adaptive', 'intelligent'],
        examples=[
            'Find the best agent for my task',
            'Route my request to available specialized agents',
            'Coordinate multiple agents for complex workflows',
        ]
    ))
    
    # Add skills from discovered agents
    for agent_id, status in agent_status.items():
        if status['available']:
            # Create a meta-skill for each available agent
            skill_id = f"delegate_to_{agent_id}"
            name = status.get('card_summary', {}).get('name') or agent_id
            skill_name = f"Delegate to {name}"
            
            dynamic_skills.append(AgentSkill(
                id=skill_id,
                name=skill_name,
                description=f"Route tasks to the {name} agent for specialized processing",
                tags=['delegation', agent_id.replace('_', '-'), 'specialized'],
                examples=[f"Use {name} for specialized tasks"]
            ))

    agent_card = AgentCard(
        name='ADK Translation Orchestrator Agent (Dynamic)',
        description=f'I orchestrate tasks across {available_count} available specialized agents including web search, code execution, content generation, and multimedia processing. I dynamically discover agent capabilities and route requests intelligently.',
        url=f'http://{host}:{port}/',
        version='2.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=dynamic_skills
    )
    
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, 
        task_store=InMemoryTaskStore()
    )
    app = A2AStarletteApplication(agent_card, request_handler)

    logger.info(f"🚀 Starting Dynamic Translation Orchestrator Agent server on http://{host}:{port}")
    logger.info(f"🔍 Agent Discovery: Complete ({available_count} agents available)")
    logger.info(f"🎯 This agent dynamically routes to available specialized agents")
    logger.info(f"📊 Agent Skills: {len(dynamic_skills)} total orchestration skills")
    
    if available_count > 0:
        logger.info("🎉 Ready to handle complex multi-agent workflows!")
    else:
        logger.warning("⚠️  Limited functionality - no sub-agents available")
    
    uvicorn.run(app.build(), host=host, port=port)

if __name__ == '__main__':
    main()