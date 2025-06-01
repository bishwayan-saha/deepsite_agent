
from typing import Annotated, Literal, Union

from models.json_rpc import JSONRPCRequest, JSONRPCResponse
from models.task import Task, TaskQueryParams, TaskSendParams
from pydantic import Field
from pydantic.type_adapter import TypeAdapter


class SendTaskRequest(JSONRPCRequest):
    method: Literal["tasks/send"] = "tasks/send"
    params: TaskSendParams 


class GetTaskRequest(JSONRPCRequest):
    method: Literal["tasks/get"] = "tasks/get"
    params: TaskQueryParams


A2ARequest = TypeAdapter(
    Annotated[
        Union[
            SendTaskRequest,
            GetTaskRequest,
        ],
        Field(discriminator="method")
    ]
)


class SendTaskResponse(JSONRPCResponse):
    result: Task | None = None


class GetTaskResponse(JSONRPCResponse):
    result: Task | None = None