import httpx, boto3, json, re
from .config import settings
from .state import GraphState
from .tool_registry import get_tool_prompt_for_intent
from .aws_session import get_bedrock_client_with_sts
from jinja2 import Template
from typing import Dict, Any

def is_tool_response(message: str) -> bool:
    return bool(re.search(r"<tool_response>.*?</tool_response>", message.strip(), re.DOTALL))

def render_chat_template(messages):
    template_str = """{{ bos_token }}{% set loop_messages = messages %}{% for message in loop_messages %}{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>{% elif message['role'] == 'tool' %}<|im_start|>tool\n{{ message['content'] }}<|im_end|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"""
    template = Template(template_str)
    rendered_prompt = template.render(messages=messages, bos_token="<|im_start|>", add_generation_prompt=True)
    return rendered_prompt

def parse_tool_call(state: GraphState) -> GraphState:
    last_msg = state["messages"][-1]
    if last_msg.get("role") != "assistant":
        return {**state, "tool_call": ""}

    content = last_msg.get("content", "")
    if not isinstance(content, str):
        return {**state, "tool_call": ""}

    match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
    if not match:
        return {**state, "tool_call": ""}

    tool_json_str = match.group(1).strip()
    return {**state, "tool_call": tool_json_str}

def classify_intent(state: GraphState) -> GraphState:
    if state.get("intent"):
        return state

    message = state["input"]
    if not is_tool_response(message):
        r = httpx.post(settings.INTENT_API_URL, json={"text": message})
        return {**state, "intent": r.json()["intent"]}
    return state

def classify_sentiment(state: GraphState) -> GraphState:
    message = state["input"]
    if not is_tool_response(message):
        r = httpx.post(settings.SENTIMENT_API_URL, json={"text": message})
        return {**state, "sentiment": r.json()["sentiment"]}
    return state

def call_travel_or_rag_api(state: GraphState) -> GraphState:
    if not state.get("tool_call"):
        return {**state, "tool_output": "No tool_call found."}
    try:
        tool = json.loads(state["tool_call"])
        name = tool.get("name")
        args = tool.get("arguments", {})
        if name == "query_policy_rag_db":
            r = httpx.post(settings.RAG_API_URL, json={"query": args.get("query", state.get("input", ""))})
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
    message = state["input"]

    # Ensure 'messages' is initialized
    if "messages" not in state:
        state["messages"] = []

    if is_tool_response(message):
        state["messages"].append({"role": "tool", "content": message.strip()})
    else:
        intent = state.get("intent", "unknown")
        sentiment = state.get("sentiment", "neutral")
        prefixed_input = f"[intent={intent}][sentiment={sentiment}] {message}"
        state["messages"].append({"role": "user", "content": prefixed_input})

    system_tool_prompt = get_tool_prompt_for_intent(state["intent"]) if "intent" in state else ""
    system_prompt = {
        "role": "system",
        "content": (system_tool_prompt or "")
    }

    full_messages = [system_prompt] + state["messages"]
    rendered_prompt = render_chat_template(full_messages)

    payload = {
        "prompt": rendered_prompt,
        "max_tokens": 512,
        "temperature": 0.3,
        "stop_sequences": ["<|im_end|>", "<|im_start|>user", "<|user|>"]
    }

    response = client.invoke_model(
        modelId=settings.BEDROCK_MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json",
        trace='ENABLED',
        guardrailIdentifier='81i8pkxk8w7l',
        guardrailVersion='1'
    )

    model_response = json.loads(response["body"].read())
    final_response = model_response["outputs"][0]["text"].strip()

    state["messages"].append({"role": "assistant", "content": final_response})

    return {**state, "messages": state["messages"]}

def append_tool_result(state: GraphState) -> GraphState:
    last_msg = state["messages"][-1]
    if last_msg["role"] == "assistant":
        content = last_msg.get("content", "")
        if "<tool_call>" in content and "</tool_call>" in content:
            tool_msg = {
                "role": "tool",
                "content": f"<tool_response>{state['tool_output']}</tool_response>"
            }
            state["messages"].append(tool_msg)
    return state
