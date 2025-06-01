
from datetime import datetime
from enum import Enum
from typing import Any, List, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


# Represents one part of a message, currently only plain text is allowed
class TextPart(BaseModel):
    type: Literal["text"] = "text"  
    text: str                     

# Alias: For now, "Part" is the same as TextPart (used for easier refactoring in the future)
Part = TextPart

# A message in the context of a task, either from the user or the agent
class Message(BaseModel):
    role: Literal["user", "agent"] 
    parts: List[Part]


# Describes the state of a task at a given moment
class TaskStatus(BaseModel):
    state: str
    timestamp: datetime = Field(default_factory=datetime.now)


# The core unit of work in the Agent2Agent protocol
class Task(BaseModel):
    id: str
    status: TaskStatus
    history: List[Message]


# Used to identify a task, e.g., when canceling or querying
class TaskIdParams(BaseModel):
    id: str
    metadata: dict[str, Any] | None = None


# Extends TaskIdParams to include optional history length
# Useful when querying a task and controlling how much of the past you want back
class TaskQueryParams(TaskIdParams):
    historyLength: int | None = None


# Parameters required to send a new task to an agent
class TaskSendParams(BaseModel):
    id: str                                
    session_id: str = Field(default_factory=lambda: uuid4().hex)
    message: Message
    historyLength: int | None = None
    metadata: dict[str, Any] | None = None


# Enum for predefined task lifecycle states
class TaskState(str, Enum):
    SUBMITTED = "submitted"           
    WORKING = "working"    
    INPUT_REQUIRED = "input-required" 
    COMPLETED = "completed" 
    CANCELED = "canceled"  
    FAILED = "failed"
    UNKNOWN = "unknown"     
    