import json
import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from models.agent import AgentCard
from models.json_rpc import InternalError, JSONRPCResponse
from models.request import A2ARequest, SendTaskRequest
from server.task_manager import TaskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class A2AServer:
    def __init__(
        self,
        host: str,
        port: int,
        agent_card: AgentCard = None,
        task_manager: TaskManager = None,
    ):
        """
        Constructor for A2AServer using FastAPI

        Args:
            host: IP address to bind the server to (default is all interfaces)
            port: Port number to listen on (default is 5000)
            agent_card: Metadata that describes our agent (name, skills, capabilities)
            task_manager: Logic to handle the task (using Gemini agent here)
        """
        self.host = host
        self.port = port
        self.agent_card = agent_card
        self.task_manager = task_manager
        self.app = FastAPI()

        @self.app.get("/.well-known/agent.json")
        async def get_agent_card():
            """Returns the agent's metadata (GET /.well-known/agent.json)"""
            return JSONResponse(self.agent_card.model_dump(exclude_none=True))

        @self.app.post("/")
        async def handle_request(request: Request):
            """
            handles task requests sent to the root path ("/").

            - 1. Parses incoming JSON
            - 2. Validates the JSON-RPC message
            - 3. For supported task types, delegates to the task manager
            - 4. Returns a response or error
            """
            try:
                # Step 1: Parse incoming JSON body
                body = await request.json()
                logger.info(f"\nIncoming JSON:\n {json.dumps(body, indent=2)}", )

                # Step 2: Parse and validate request using discriminated union
                json_rpc = A2ARequest.validate_python(body)

                # Step 3: If it’s a send-task request, call the task manager to handle it
                if isinstance(json_rpc, SendTaskRequest):
                    result = await self.task_manager.on_send_task(json_rpc)
                else:
                    raise ValueError(f"Unsupported A2A method: {type(json_rpc)}")

                # Step 4: Convert the result into a proper JSON response
                return self.create_response(result)

            except Exception as e:
                logger.error(f"Error occurred while delegating task: \n Reason: {e}")
                return JSONResponse(
                    JSONRPCResponse(id=None, error=InternalError(message=str(e))).model_dump(),
                    status_code=400,
                )

    def create_response(self, result):
        """
        Converts a JSONRPCResponse object into a JSON HTTP response.

        Args:
            result: The response object (must be a JSONRPCResponse)

        Returns:
            JSONResponse: Starlette-compatible HTTP response with JSON body
        """
        if isinstance(result, JSONRPCResponse):
            return JSONResponse(
                content=jsonable_encoder(result.model_dump(exclude_none=True))
            )
        else:
            raise ValueError("Invalid response type")

    def start(self):
        """Starts the A2A server using uvicorn."""
        if not self.agent_card or not self.task_manager:
            raise ValueError("Agent card and task manager are required")
        uvicorn.run(self.app, host=self.host, port=self.port, timeout_keep_alive=150)
