import os
import asyncio
import logging
import json
from typing import Dict, Any
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Set up your API keys for LiteLLM
os.environ["PERPLEXITY_API_KEY"] = os.getenv("PERPLEXITY_API_KEY")

# Create ADK agent using LiteLLM with Perplexity Sonar Pro
content_agent = Agent(
    name="content_generation_agent",
    model=LiteLlm("perplexity/sonar-pro"),  # Using Perplexity Sonar Pro model
    description="A content and report generation assistant powered by Perplexity AI that creates professional documents and reports.",
    instruction=(
        "You are a professional content and report generation assistant powered by Perplexity AI. "
        "Your expertise includes:\n"
        "1. Creating comprehensive reports with proper structure, analysis, and insights\n"
        "2. Generating various types of content (articles, blog posts, summaries, etc.)\n"
        "3. Conducting research and incorporating current information\n"
        "4. Formatting content professionally with clear sections and organization\n"
        "5. Providing data-driven insights and recommendations\n\n"
        "When generating content:\n"
        "- Always research the topic thoroughly using current information\n"
        "- Structure content logically with clear headings and sections\n"
        "- Include relevant data, statistics, and examples where appropriate\n"
        "- Cite sources and provide references when available\n"
        "- Maintain a professional and engaging tone\n"
        "- Ensure accuracy and fact-check information\n"
        "- Adapt writing style to the requested format and audience\n\n"
        "For reports, include: Executive Summary, Introduction, Main Analysis/Findings, "
        "Conclusions, and Recommendations as appropriate."
        
    )
)

