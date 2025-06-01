import logging

import click

from agent.agent import DeepSiteAgent
from agent.task_manager import RedditAgentTaskManager
from models.agent import AgentCapabilities, AgentCard, AgentSkill
from server.server import A2AServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="localhost", help="Host to bind the server to")
@click.option("--port", default=10003, help="Port number for the server")
def main(host, port):
    """
    This function sets up everything needed to start the agent server.
    """
    logger.info(" --- DeepSite Agent Started --- ")

    # Define what this agent can do – in this case, it does NOT support streaming
    capabilities = AgentCapabilities(streaming=False)

    # Define the skill this agent offers (used in directories and UIs)
    skill = AgentSkill(
        id="generate_ui_code",
        name="Genearte UI Code in HTML CSS JavaScript using DeepSite API",
        description="Given a user prompt, it uses the DeepSite API to generate HTML code and returns that code in the HTML format",  # What the skill does
        tags=["ui", "html", "frontend"],
        examples=[
            "Generate a navbar with buttons having shadow.",
            "Create a UI loading screen",
        ],
    )

    # Create an agent card describing this agent's identity and metadata
    agent_card = AgentCard(
        name="DeepSiteAgent",
        description="This agent generates HTML code given a user prompt.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=DeepSiteAgent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes=DeepSiteAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )

    server = A2AServer(
        host=host,
        port=port,
        agent_card=agent_card,
        task_manager=RedditAgentTaskManager(agent=DeepSiteAgent()),
    )

    server.start()


if __name__ == "__main__":
    main()
