import httpx, boto3, json, re
from .config import settings
from .state import GraphState
from .tool_registry import get_tool_prompt_for_intent
from .aws_session import get_bedrock_client_with_sts

def parse_tool_call(state: GraphState) -> GraphState:
    """Extracts <tool_call>{...}</tool_call> block from the assistant's response"""
    
    last_msg = state["messages"][-1]

    # ✅ Check if this message was from the assistant
    if last_msg.get("role") != "assistant":
        return {**state, "tool_call": ""}

    content = last_msg.get("content", "")
    if not isinstance(content, str):
        return {**state, "tool_call": ""}

    # ✅ Extract tool call block using regex
    match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
    if not match:
        return {**state, "tool_call": ""}

    tool_json_str = match.group(1).strip()

    return {**state, "tool_call": tool_json_str}

def classify_intent(state: GraphState) -> GraphState:
    r = httpx.post(settings.INTENT_API_URL, json={"text": f"{state['messages']} {state['input']}"})
    return {**state, "intent": r.json()["intent"]}

def classify_sentiment(state: GraphState) -> GraphState:
    r = httpx.post(settings.SENTIMENT_API_URL, json={"text": state["input"]})
    return {**state, "sentiment": r.json()["sentiment"]}

def call_travel_or_rag_api(state: GraphState) -> GraphState:
    if not state.get("tool_call"):
        return {**state, "tool_output": "No tool_call found."}

    try:
        tool = json.loads(state["tool_call"])
        name = tool.get("name")
        args = tool.get("arguments", {})

        # Route to appropriate external service
        if name == "query_policy_rag_db":
            r = httpx.post(settings.RAG_API_URL, json={"query": state["input"]})
            return {**state, "tool_output": r.json()["result"]}
            
        elif name == "search_flight":
            r = httpx.post(settings.SEARCH_FLIGHT_API_URL, json=args)
            return {**state, "tool_output": r.json()["data"]}
         
        elif name == "check_flight_offers":
            r = httpx.post(settings.CHECK_FLIGHT_OFFERS_API_URL, json=args)
            return {**state, "tool_output": r.json()["data"]}

        elif name == "book_flight":
            r = httpx.post(settings.BOOK_FLIGHT_API_URL, json=args)
            return {**state, "tool_output": r.json()["data"]}

        elif name == "check_baggage_allowance":
            r = httpx.get(settings.BAGGAGE_STATUS_API_URL, params=args)
            return {**state, "tool_output": r.json()["data"]}

        else:
            return {**state, "tool_output": f"Unknown tool: {name}"}

    except Exception as e:
        return {**state, "tool_output": f"Tool call parsing failed: {str(e)}"}


def call_bedrock_model(state: GraphState) -> GraphState:
    client = get_bedrock_client_with_sts()

    # Compose user input with intent/sentiment prefix
    prefixed_input = f"[intent={state['intent']}][sentiment={state['sentiment']}] {state['input']}"
       
    state["messages"].append({
        "role": "user",
        "content": prefixed_input
    })


    # Retrieve the system prompt for the intent
    system_tool_prompt = get_tool_prompt_for_intent(state["intent"])
    context = state.get("tool_output")
    
    system_prompt = {
      "role": "system",
      "content": (system_tool_prompt or "") + "\n\nContext: " + (context or "")
    }
    
    # Construct wrapped payload exactly like in playground
    bedrock_prompt_payload = {
        "messages": state["messages"]
    }

    payload = {
        "prompt": json.dumps(bedrock_prompt_payload),
        "max_tokens": 384,
        "temperature": 0.5
    }
    
    # Invoke model
    response = client.invoke_model(
        modelId=settings.BEDROCK_MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json"
    )

    model_response = json.loads(response["body"].read())

    # Extract final response from first content block
    final_response = model_response["outputs"][0]["text"]
    
    state["messages"].append({
        "role": "assistant",
        "content": final_response.strip()
    })

    return {
        **state,
        "messages": state["messages"]
    }



def append_tool_result(state: GraphState) -> GraphState:
    # Get last assistant message
    last_msg = state["messages"][-1]

    # Check if it's an assistant response and has <tool_call> tag
    if last_msg["role"] == "assistant":
        content = last_msg.get("content", "")
        
        # Use regex or basic check for <tool_call> tag
        if "<tool_call>" in content and "</tool_call>" in content:
            tool_msg = {
                "role": "tool",
                "content": f"<tool_response>{state['tool_output']}</tool_response>"
            }
            state["messages"].append(tool_msg)

    return state
