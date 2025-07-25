# app/router.py

from fastapi import APIRouter, Request, Depends
from .graph import WorkflowFactory
from .config import settings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize the workflow factory instance
workflow_factory = WorkflowFactory(redis_url=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}")

# Dependency to get compiled workflow
def get_compiled_workflow():
    return workflow_factory.get_workflow()

@router.get("/health")
async def health_check():
    # Lightweight check for ALB
    return {"status": "ok", "service": "new langgraph api"}


@router.post("/chat")
async def chat_endpoint(request: Request, workflow=Depends(get_compiled_workflow)):
    try:
        data = await request.json()

        # Get last message (user or tool)
        last_message = data.get("message")
        if not last_message or not isinstance(last_message, dict):
            return {"error": "Missing or invalid 'message' object."}

        # Validate content
        content = last_message.get("content", "").strip()
        if not content:
            return {"error": "Message content is empty."}

        # Validate role
        role = last_message.get("role", "").strip()
        if role not in ["user", "tool"]:
            return {"error": "Invalid message role. Must be 'user' or 'tool'."}

        # Get or default the session ID
        session_id = data.get("session_id", "anonymous-session")

        # Invoke workflow — LangGraph RedisSaver will restore message history
        result = workflow.invoke(
            input={
                "input": content,
                "role": role
            },
            config={
                "configurable": {
                    "session_id": session_id,  # Optional: for your tracking
                    "thread_id": session_id    # Required for RedisSaver
                }
            }
        )

        return {"messages": result["messages"]}

    except Exception as e:
        logger.error(f"[LangGraph Error] {str(e)}", exc_info=True)
        return {
            "error": "Sorry for the inconvenience, we are unable to process the request at this moment. Please try again later."
        }
