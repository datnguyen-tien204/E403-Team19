from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator


class SearchModeState:
    def __init__(self, mode: str = "hybrid"):
        self.mode = mode

    def set_mode(self, mode: str):
        mode = (mode or "").lower()
        if mode not in ("hybrid", "on", "off"):
            mode = "hybrid"
        self.mode = mode

    def get_mode(self) -> str:
        return self.mode


class AgentState(TypedDict):
    # Toàn bộ lịch sử chat (HumanMessage + AIMessage + ToolMessage)
    messages: Annotated[Sequence[BaseMessage], operator.add]
