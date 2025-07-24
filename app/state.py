from typing import List, Dict, TypedDict

class GraphState(TypedDict):
    input: str
    intent: str
    sentiment: str
    messages: List[Dict[str, str]]
    tool_call: str  # Raw XML or JSON extracted
    tool_output: str
