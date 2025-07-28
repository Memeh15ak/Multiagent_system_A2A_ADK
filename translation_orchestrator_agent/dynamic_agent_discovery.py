# simplified_agent_discovery.py - Extract full agent cards for ADK routing
import asyncio
import httpx
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class AgentCardInfo:
    """Complete agent card information extracted from health endpoint."""
    agent_id: str
    url: str
    available: bool = False
    full_card: Optional[Dict[str, Any]] = None
    raw_response: Optional[Dict[str, Any]] = None

class SimpleAgentCardDiscovery:
    """Extract complete agent cards from health endpoints for ADK routing."""
    
    def __init__(self, agent_registry: Dict[str, str] = None):
        """Initialize with agent registry."""
        self.agent_registry = agent_registry or {
            "web_searcher": "http://localhost:10012",
            "code_executor": "http://localhost:10011", 
            "audio_processor": "http://localhost:10013",
            "img_to_img_processor": "http://localhost:10014",
            "report_content_generator": "http://localhost:10015",
            "excel_file_executor": "http://localhost:10016",
            "video_processor": "http://localhost:10021",
            "rag_agent": "http://localhost:10019",
            "image_analyzer": "http://localhost:10017",
            "image_generation": "http://localhost:10018"

        }
        self.agent_cards: Dict[str, AgentCardInfo] = {}
        self.discovery_timeout = 10.0
        
    async def extract_all_agent_cards(self) -> Dict[str, AgentCardInfo]:
        """Extract complete agent cards from all registered agents."""
        logger.info("Extracting agent cards from health endpoints...")
        
        # Initialize agent card info objects
        for agent_id, agent_url in self.agent_registry.items():
            self.agent_cards[agent_id] = AgentCardInfo(agent_id=agent_id, url=agent_url)
        
        # Extract cards concurrently
        tasks = [
            self._extract_single_agent_card(card_info) 
            for card_info in self.agent_cards.values()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        available_agents = [c.agent_id for c in self.agent_cards.values() if c.available]
        unavailable_agents = [c.agent_id for c in self.agent_cards.values() if not c.available]
        
        logger.info(f"Card extraction complete. Available: {available_agents}")
        if unavailable_agents:
            logger.warning(f"Unavailable: {unavailable_agents}")
            
        return self.agent_cards
    
    async def _extract_single_agent_card(self, card_info: AgentCardInfo) -> None:
        """Extract agent card from a single agent's health endpoint."""
        try:
            async with httpx.AsyncClient(timeout=self.discovery_timeout) as client:
                
                # Try health endpoint first (most common)
                endpoints_to_try = [
                    f"{card_info.url}/health",
                    f"{card_info.url}",
                    f"{card_info.url}/agent",
                    f"{card_info.url}/card"
                ]
                
                for endpoint in endpoints_to_try:
                    try:
                        response = await client.get(endpoint)
                        if response.status_code == 200:
                            response_data = response.json()
                            
                            # Store raw response for debugging
                            card_info.raw_response = response_data
                            
                            # Extract the complete agent card
                            agent_card = self._extract_agent_card_from_response(response_data)
                            
                            if agent_card:
                                card_info.full_card = agent_card
                                card_info.available = True
                                logger.info(f"✅ Extracted agent card for {card_info.agent_id} from {endpoint}")
                                return
                                
                    except Exception as e:
                        logger.debug(f"Failed to get card from {endpoint} for {card_info.agent_id}: {e}")
                        continue
                
                # If all endpoints fail
                card_info.available = False
                logger.warning(f"❌ Could not extract agent card for {card_info.agent_id}")
                
        except Exception as e:
            logger.error(f"Error extracting card for {card_info.agent_id}: {e}")
            card_info.available = False
    
    def _extract_agent_card_from_response(self, response_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract agent card from API response, preserving complete structure."""
        try:
            # Look for agent card in different possible locations
            possible_card_keys = ['agent_card', 'card', 'agent', 'agent_info']
            
            agent_card = None
            for key in possible_card_keys:
                if key in response_data and isinstance(response_data[key], dict):
                    agent_card = response_data[key]
                    break
            
            # If no nested card found, assume the whole response is the card
            if not agent_card and self._looks_like_agent_card(response_data):
                agent_card = response_data
            
            if agent_card:
                # Ensure it has minimum required fields
                if not agent_card.get('name'):
                    agent_card['name'] = agent_card.get('id', 'Unknown Agent')
                    
                if not agent_card.get('description'):
                    agent_card['description'] = 'No description provided'
                    
                return agent_card
                
            return None
            
        except Exception as e:
            logger.error(f"Error extracting agent card from response: {e}")
            return None
    
    def _looks_like_agent_card(self, data: Dict[str, Any]) -> bool:
        """Check if data structure looks like an agent card."""
        # Check for common agent card fields
        card_indicators = ['name', 'description', 'skills', 'capabilities', 'version']
        return any(key in data for key in card_indicators)
    
    def generate_adk_prompt_with_agent_cards(self, user_query: str) -> str:
        """Generate ADK prompt with complete agent cards for intelligent routing."""
        
        available_cards = [card for card in self.agent_cards.values() if card.available and card.full_card]
        
        if not available_cards:
            return f"""
USER QUERY: {user_query}

❌ NO AGENTS AVAILABLE
No agents are currently running. Please start the required agents and try again.
"""
        
        prompt_parts = [
            f"USER QUERY: {user_query}",
            "",
            "🤖 AVAILABLE AGENTS AND THEIR COMPLETE CAPABILITIES:",
            "=" * 60,
            ""
        ]
        
        # Function mapping for routing
        function_mapping = {
            "web_searcher": "web_search_function",
            "code_executor": "code_execution_function", 
            "audio_processor": "audio_conversational_function",
            "img_to_img_processor": "image_modification_function",
            "video_processor": "video_function",
            "report_content_generator": "report_and_content_generation_function",
            "excel_file_analysis" : "excel_file_analysis_function",
            "rag_agent": "rag_agent_function",
            "image_analyzer": "image_analyzer_function",
            "image_generation": "image_generation_function"    
        }
        
        for card_info in available_cards:
            agent_card = card_info.full_card
            function_name = function_mapping.get(card_info.agent_id, f"{card_info.agent_id}_function")
            
            prompt_parts.extend([
                f"🔹 AGENT: {agent_card.get('name', card_info.agent_id)}",
                f"   ID: {card_info.agent_id}",
                f"   Function to call: {function_name}",
                f"   URL: {card_info.url}",
                ""
            ])
            
            # Add complete agent card as JSON for full context
            prompt_parts.extend([
                "   📋 COMPLETE AGENT CARD:",
                "   ```json"
            ])
            
            # Pretty print the agent card
            card_json = json.dumps(agent_card, indent=2)
            for line in card_json.split('\n'):
                prompt_parts.append(f"   {line}")
            
            prompt_parts.extend([
                "   ```",
                "",
                "   " + "─" * 50,
                ""
            ])
        
        prompt_parts.extend([
            "",
            "🎯 ROUTING INSTRUCTIONS:",
            "1. Analyze the user query carefully",
            "2. Review each agent's complete card (skills, capabilities, examples)",
            "3. Select the MOST APPROPRIATE agent based on:",
            "   • Skills and their descriptions",
            "   • Examples provided in skills",
            "   • Input/output capabilities",
            "   • Agent description and purpose",
            "4. Call the corresponding function ONCE",
            "5. Return the COMPLETE response from the agent",
            "",
            "🔥 CRITICAL RULES:",
            "• ONE function call per user query",
            "• Use the exact function name specified above",
            "• Pass the original user query to the function",
            "• Return the FULL response from the agent (no summarization)",
            "• If function fails, explain and suggest alternatives",
            "",
            f"NOW ROUTE THIS QUERY: {user_query}"
        ])
        
        return "\n".join(prompt_parts)
    
    def get_available_agents_summary(self) -> str:
        """Get a quick summary of available agents."""
        available = [c for c in self.agent_cards.values() if c.available]
        unavailable = [c for c in self.agent_cards.values() if not c.available]
        
        summary = [
            f"📊 AGENT STATUS SUMMARY:",
            f"✅ Available: {len(available)} agents",
            f"❌ Unavailable: {len(unavailable)} agents",
            ""
        ]
        
        if available:
            summary.append("Available agents:")
            for card in available:
                name = card.full_card.get('name', card.agent_id) if card.full_card else card.agent_id
                summary.append(f"  • {name} ({card.agent_id})")
        
        if unavailable:
            summary.append("\nUnavailable agents:")
            for card in unavailable:
                summary.append(f"  • {card.agent_id}")
        
        return "\n".join(summary)
    
    def get_agent_card(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get the complete agent card for a specific agent."""
        card_info = self.agent_cards.get(agent_id)
        return card_info.full_card if card_info and card_info.available else None


# Example usage function
async def main():
    """Example of how to use the simplified agent card discovery."""
    discovery = SimpleAgentCardDiscovery()
    
    # Extract all agent cards
    agent_cards = await discovery.extract_all_agent_cards()
    
    # Print summary
    print(discovery.get_available_agents_summary())
    
    # Example user query
    user_query = "Search for latest AI news and create a summary report"
    
    # Generate ADK prompt with complete agent cards
    adk_prompt = discovery.generate_adk_prompt_with_agent_cards(user_query)
    print("\n" + "="*80)
    print("ADK PROMPT WITH AGENT CARDS:")
    print("="*80)
    print(adk_prompt)

if __name__ == "__main__":
    asyncio.run(main())