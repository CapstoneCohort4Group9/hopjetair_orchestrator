from fastapi import FastAPI
from .router import router
from .graph import WorkflowFactory
from .config import settings

app = FastAPI(
    title="LLM Orchestrator",
    description="Orchestrates LLM-based travel chatbot using LangGraph",
    version="1.0.0"
)

# Initialize the workflow factory and run setup on startup
workflow_factory = WorkflowFactory(redis_url=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}")

@app.on_event("startup")
async def on_startup():
    workflow_factory.setup()

app.include_router(router)
