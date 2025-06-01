from agent.agent import DeepSiteAgent
from models.request import SendTaskRequest, SendTaskResponse
from models.task import Message, TaskStatus, TextPart
from server.task_manager import InMemoryTaskManager
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedditAgentTaskManager(InMemoryTaskManager):
    """
    This class connects the agent to task system.
    - It inherits all the logics from 'InMemoryTaskManager'
    - It also implements the on_send_task method which is used to handle a new task
    """

    def __init__(self, agent: DeepSiteAgent):
        super().__init__()
        self.agent = agent

    # Extract user's query from incoming task request
    def _get_user_query(self, request: SendTaskRequest) -> str:
        """
        Extract the user's query in string format from the task request
        Args:
            request (SendTaskRequest): The task request containing user's query
        Returns:
            str: The actual text in string within user's query object
        """
        return request.params.message.parts[0].text

    # Main logic to handle and execute a task
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        1. Save or update the task into memory
        2. Ask the agent for a response
        3. Fornat the reply as a message object
        4. Save this response into task history
        5. Return the updated task to the user/caller
        """

        # 1. Save or update the task into memory
        task = await self.upsert_task(request.params)

        # 2. Ask the agent for a response
        response_text = await self.agent.invoke(
            self._get_user_query(request), request.params.session_id
        )

        # 3. Format the reply as a message object
        agent_response = Message(role="agent", parts=[TextPart(text=response_text)])
        logger.info(f"\nOutgoing JSON Response:\n {json.dumps(agent_response.model_dump(), indent=2)}")

        # 4. Save this response into task history
        async with self.lock:
            task.status = TaskStatus(state="COMPLETED")
            task.history.append(agent_response)

        # 5. Return the updated task to the user/caller
        return SendTaskResponse(result=task)
