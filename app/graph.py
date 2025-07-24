# app/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver
from .config import settings
from .state import GraphState
from .nodes import (
    classify_intent,
    classify_sentiment,
    call_travel_or_rag_api,
    parse_tool_call,
    append_tool_result,
    call_bedrock_model
)
import redis


class WorkflowFactory:
    def __init__(self, redis_url: str):
        self.redis_client = redis.Redis.from_url(redis_url)
        self.saver = RedisSaver(
            redis_client=self.redis_client,
            ttl={
                "default_ttl": 3600 * 24 * 5,     # 5 days TTL in seconds
                "refresh_on_read": True           # Optional: resets TTL on every access
            }
        )


    def setup(self):
        self.saver.setup()  # Ensure required indices are created

    def get_workflow(self):
        wf = StateGraph(GraphState)

        wf.add_node("intent_classifier", classify_intent)
        wf.add_node("sentiment_classifier", classify_sentiment)
        wf.add_node("bedrock_inference", call_bedrock_model)
        wf.add_node("parse_tool_call", parse_tool_call)
        wf.add_node("call_tool", call_travel_or_rag_api)
        wf.add_node("append_tool_output", append_tool_result)

        wf.set_entry_point("intent_classifier")
        wf.add_edge("intent_classifier", "sentiment_classifier")
        wf.add_edge("sentiment_classifier", "bedrock_inference")
        wf.add_edge("bedrock_inference", "parse_tool_call")
        wf.add_edge("parse_tool_call", "call_tool")
        wf.add_edge("call_tool", "append_tool_output")
        wf.add_edge("append_tool_output", END)

        return wf.compile(checkpointer=self.saver)
