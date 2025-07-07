from fastapi import FastAPI
from .router import router

app = FastAPI(
    title="LLM Orchestrator",
    description="Orchestrates LLM-based travel chatbot using LangGraph",
    version="1.0.0"
)

app.include_router(router)
