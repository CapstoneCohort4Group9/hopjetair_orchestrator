from fastapi import APIRouter, Request, HTTPException
from .graph import get_workflow

router = APIRouter()
workflow = get_workflow()

@router.post("/chat")
async def chat_endpoint(request: Request):
     try:
        data = await request.json()
        user_input = data.get("message")
        messages = data.get("messages", [])

        # Run the LangGraph workflow
        result = workflow.invoke({
            "input": user_input,
            "messages": messages
        })

        return {
            "messages": result["messages"]
        }

     except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
