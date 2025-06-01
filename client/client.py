import json
from typing import Any
from uuid import uuid4

import httpx

from models.agent import AgentCard
from models.json_rpc import JSONRPCRequest
from models.request import SendTaskRequest
from models.task import Task, TaskSendParams


class A2AClientHTTPError(Exception):
    """Raised when an HTTP request fails (e.g., bad server response)"""

    pass


class A2AClientJSONError(Exception):
    """Raised when the response is not valid JSON"""

    pass


class A2AClient:
    def __init__(self, agent_card: AgentCard = None, url: str = None):
        if agent_card:
            self.url = agent_card.url
        elif url:
            self.url = url
        else:
            raise ValueError("Either agent card or url must be provided")

    async def send_task(self, payload: dict[str, Any]):
        request = SendTaskRequest(id=uuid4().hex, params=TaskSendParams(**payload))

        print("\n----- Sending JSON RPC request -----\n")

        response = await self._send_request(request)
        return Task(**response["result"])

    async def _send_request(self, request: JSONRPCRequest) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.url, json=request.model_dump(), timeout=150
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                raise A2AClientHTTPError(e.response.status_code, str(e)) from e

            except json.JSONDecodeError as e:
                raise A2AClientJSONError(str(e)) from e
