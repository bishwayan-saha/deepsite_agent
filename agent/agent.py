import logging
import os

import requests
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import os
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

async def get_code_from_deepsite(prompt: str):
    """
    Asynchronously fetches code generation output from the DeepSite API.
    Args:
        prompt (str): The user-provided prompt to generate frontend code.
    Returns:
        str: The response text from the DeepSite API.
    Raises:
        Exception: Logs an error if the request fails.
    Notes:
        - Uses an HF token from environment variables for authentication.
        - Sends a JSON payload with the prompt and provider information.
        - Implements a timeout of 150 seconds to handle API latency.
    """

    url = "https://enzostvs-deepsite.hf.space/api/ask-ai"
    headers = {
        "Cookie": f"hf_token={os.getenv("HUGGINGFACE_API_KEY")}",
        "Content-Type": "application/json",
    }
    data = {
        "prompt": prompt,
        "provider": "auto",
        "model": "deepseek-ai/DeepSeek-V3-0324",
    }
    try:
        response = await httpx.AsyncClient().post(
            url, headers=headers, json=data, timeout=150
        )
    except Exception as e:
        logger.error(f"Error occurred while fetching deepsite agent \n Reason {e}")
    return response.text


class DeepSiteAgent:

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        """
        Initialize the DeepSiteAgent.
        Sets up session handling, memory and runner to execute task
        """
        server_domain = os.getenv("SERVER_DOMAIN") or "http://localhost"
        logger.info(f"Server domain for calling credentials: {server_domain}")
        self._credentials = requests.get(
            f"{server_domain}:3100/credentials"
        ).json()
        for creds in self._credentials["data"]:
            os.environ[creds] = self._credentials["data"].get(creds)
        load_dotenv()

        self._agent = self._build_agent()
        self._user_id = "remote_reddit_agent"
        ## Runner is used to manage the agent and its environment
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),  # retreive files/ docs/ artifacts
            session_service=InMemorySessionService(),  # keeps track of conversation by managing sessions
            memory_service=InMemoryMemoryService(),  # Optional: remembers past messages
        )

    def _build_agent(self) -> Agent:
        """Creates and returns an LlmAgent instance"""
        return Agent(
            model="gemini-2.0-flash",
            name="deepsite_agent",
            description="""An AI agent which takes help of DeepSite api to generates UI code given a user prompt.""",
            instruction="""
                            ### Website UI Generator using DeepSite API
                            Your task is to generate frontend code by calling the **`get_code_from_deepsite`** tool.

                            #### **Steps:**
                            1. **Identify Intent:** Ensure the request is related to UI or website design.
                            2. **Extract User Prompt:** Use the user’s input to determine the request.
                            3. **Generate Output:** Retrieve **HTML code only** from the tool.
                            4. **Mandatory Tool Call:** You **MUST** invoke `get_code_from_deepsite`.  
                            - **Do NOT** generate code manually.  
                            - **Do NOT** hallucinate responses.  

                            ⚠️ Always rely on the tool for accurate code generation.                
                        """,
            tools=[get_code_from_deepsite],
        )

    async def invoke(self, query: str, session_id: str) -> str:
        """
        Receives a user query about designing an UI and returns a htlm response
        Args:
            query (str): The user prompt to be processed
            session_id (str): The session ID for context of grouping messages
        Returns:
            str: The htlm code response from the agent
        """
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name, user_id=self._user_id, session_id=session_id
        )
        print(f"Hello 1 {session}")
        if not session:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name, user_id=self._user_id, session_id=session_id
            )
        print(f"Hello 2 {session}")
        ## Formatting user message in way the model can understand
        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        print(f"Hello 3 {content}")
        ## Run the aget using Runner and get the response events
        events = list(
            self._runner.run(
                user_id=self._user_id, session_id=session_id, new_message=content
            )
        )
        print(f"Hello 4 {events}")
        ## Fallback response if no events are returned
        if not events or not events[-1].content or not events[-1].content.parts:
            return "No response from agent"

        ## Extract the responses text from all events and join them
        response_text = "\n".join(
            [part.text for part in events[-1].content.parts if part.text]
        )

        return response_text
