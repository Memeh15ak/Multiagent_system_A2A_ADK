# 🧠 Multi-Agent System via A2A Protocol (Google ADK)

A modular, multi-agent system leveraging the **A2A (Agent-to-Agent) protocol** over **Google ADK**.  
It includes agents for code execution, web search, report generation, image processing, translation orchestration, dynamic discovery, and a centralized orchestrator that routes queries based on agent discovery cards.

---

## 📂 Project Structure

MAS/
│
├── Agents/
│ ├── audio_agent/
│ ├── code_executor_agent/
│ ├── img_to_img_agent/
│ ├── report_gen_agent/
│ ├── web_search_agent/
│ ├── test.py # Test utilities
│ └── init.py
│
├── translation_orchestrator/
│ ├── main.py # Translation orchestrator agent entrypoint
│ └── init.py
│
├── adk_agent_executor.py # ADK runner for agents
├── adk_agent.py # Custom ADK agent class definitions
├── cli_client.py # CLI client for querying orchestrator agent
├── demo.py # Demo script for agent orchestration
├── dynamic_agent_discovery.py # Dynamic agent card discovery and routing
├── tools.py # Utility functions and tools used by orchestrator
│
├── .env # Local environment variables (ignored via .gitignore)
├── .gitignore
├── requirements.txt
└── README.md

ruby
Copy
Edit

---

## 🛠️ Agents and Roles

| Agent                        | Description                                                     | Run Command                                              |
|:----------------------------|:----------------------------------------------------------------|:----------------------------------------------------------|
| `code_executor_agent`        | Executes code snippets dynamically via orchestrator.            | `python -m Agents.code_executor_agent.__main__`          |
| `web_search_agent`           | Queries web search APIs and returns results.                     | `python -m Agents.web_search_agent.__main__`             |
| `report_gen_agent`           | Generates reports based on structured/unstructured data.         | `python -m Agents.report_gen_agent.__main__`             |
| `img_to_img_agent`           | Processes or transforms images.                                  | `python -m Agents.img_to_img_agent.__main__`             |
| `audio_agent`                | (Assumed for audio file analysis, not in CLI yet)                 | —                                                        |
| `translation_orchestrator`   | Handles translation tasks and routes to translation agents.       | `python -m translation_orchestrator.__main__`            |

---

## 📡 Orchestrator Agent (Main Controller)

Central orchestrator that:
- Discovers agent cards dynamically  
- Routes incoming queries and multimedia requests to appropriate agents  
- Maintains a registry of available agents via **A2A protocol**

**Run:**
```bash
python -m orchestrator_agent.__main__
📡 CLI Client
A command-line client to send queries and multimedia files to the orchestrator.

Usage:

bash
Copy
Edit
python -m orchestrator_agent.cli_client --query "your query" --multimedia_file "path/to/file"
Example:

bash
Copy
Edit
python -m orchestrator_agent.cli_client --query "translate image" --multimedia_file "media/image_samples/sample_chart.png"
🛠️ Supporting Modules
adk_agent_executor.py → Manages launching agents via Google ADK

adk_agent.py → Defines agent classes and base functionalities

dynamic_agent_discovery.py → Discovers agent cards at runtime from /well-known/agents.json

tools.py → Common utility functions for orchestrator logic and agent calls

demo.py → Script to test orchestration and agent chaining

📝 Environment
All environment variables (API keys etc.) are placed in .env (excluded from git)

Python dependencies listed in requirements.txt

📊 Media & Report Storage
media/: For input images, audio, videos

reports/: For generated reports

📸 Screenshots
You may add screenshots like this:


📦 Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
✅ To Run Entire System
1️⃣ Start the orchestrator:

bash
Copy
Edit
python -m orchestrator_agent.__main__
2️⃣ Start individual agents:

bash
Copy
Edit
python -m Agents.code_executor_agent.__main__
python -m Agents.web_search_agent.__main__
python -m Agents.report_gen_agent.__main__
python -m Agents.img_to_img_agent.__main__
3️⃣ Use the CLI client:

bash
Copy
Edit
python -m orchestrator_agent.cli_client --query "example" --multimedia_file "path/to/file"
📣 Notes
Ensure .env is configured before running

Agents communicate over A2A using well-known JSON agent cards

Supports text, image, audio, video inputs via CLI client

📌 Credits
Created by Mehak Arora 🐍✨
Multi-Agent System, Agentic AI experiments powered by Google ADK.
