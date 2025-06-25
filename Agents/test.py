from anthropic import Anthropic
import os
from dotenv import load_dotenv

import anthropic
load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# Initialize with both beta headers
client = Anthropic(
    default_headers={
        "anthropic-beta": "code-execution-2025-05-22,files-api-2025-04-14"
    }
)

# Request code execution that creates files
response = client.messages.create(
    model="claude-opus-4-20250514",
    max_tokens=4096,
    messages=[{
        "role": "user",
        "content": "Create a matplotlib visualization and save it as output.png"
    }],
    tools=[{
        "type": "code_execution_20250522",
        "name": "code_execution"
    }]
)
# Extract file IDs from the response
def extract_file_ids(response):
    file_ids = []
    for item in response.content:
        if item.type == 'code_execution_tool_result':
            content_item = item.content
            if content_item.get('type') == 'code_execution_result':
                for file in content_item.get('content', []):
                    file_ids.append(file['file_id'])
    return file_ids

# Download the created files
for file_id in extract_file_ids(response):
    file_metadata = client.beta.files.retrieve_metadata(file_id)
    file_content = client.beta.files.download(file_id)
    file_content.write_to_file(file_metadata.filename)
    print(f"Downloaded: {file_metadata.filename}")