import httpx, boto3, json, re
from .config import settings
from .state import GraphState
from .tool_registry import get_tool_prompt_for_intent
from urllib.parse import urljoin
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

# RAG-based tool names (POST calls using query string)
RAG_INTENTS = {
    "check_baggage_allowance",
    "check_cancellation_fee",
    "check_flight_insurance_coverage",
    "search_flight_insurance",
    "check_trip_insurance_coverage",
    "search_trip_insurance",
    "query_policy_rag_db"
}

INTENT_ROUTING_MAP = {
    "book_activity": settings.BOOK_ACTIVITY_API_URL,
    "book_excursion": settings.BOOK_EXCURSION_API_URL,
    "book_flight": settings.BOOK_FLIGHT_API_URL,
    "book_trip": settings.BOOK_TRIP_API_URL,
    "cancel_booking": settings.CANCEL_BOOKING_API_URL,
    "cancel_trip": settings.CANCEL_TRIP_API_URL,
    "change_flight": settings.CHANGE_FLIGHT_API_URL,
    "change_seat": settings.CHANGE_SEAT_API_URL,
    "change_trip": settings.CHANGE_TRIP_API_URL,
    "check_arrival_time": settings.CHECK_ARRIVAL_TIME_API_URL,
    "check_departure_time": settings.CHECK_DEPARTURE_TIME_API_URL,
    "check_excursion_availability": settings.CHECK_EXCURSION_AVAILABILITY_API_URL,
    "check_flight_availability": settings.CHECK_FLIGHT_AVAILABILITY_API_URL,
    "check_flight_availability_and_fare": settings.CHECK_FLIGHT_AVAILABILITY_AND_FARE_API_URL,
    "check_flight_checkin_status": settings.CHECK_FLIGHT_CHECKIN_STATUS_API_URL,
    "check_flight_offers": settings.CHECK_FLIGHT_OFFERS_API_URL,
    "check_flight_prices": settings.CHECK_FLIGHT_PRICES_API_URL,
    "check_flight_reservation": settings.CHECK_FLIGHT_RESERVATION_API_URL,
    "check_flight_status": settings.CHECK_FLIGHT_STATUS_API_URL,
    "check_in": settings.CHECK_IN_API_URL,
    "check_in_passenger": settings.CHECK_IN_PASSENGER_API_URL,
    "check_refund_eligibility": settings.CHECK_REFUND_ELIGIBILITY_API_URL,
    "check_seat_availability": settings.CHECK_SEAT_AVAILABILITY_API_URL,
    "check_trip_details": settings.CHECK_TRIP_DETAILS_API_URL,
    "check_trip_offers": settings.CHECK_TRIP_OFFERS_API_URL,
    "check_trip_plan": settings.CHECK_TRIP_PLAN_API_URL,
    "check_trip_prices": settings.CHECK_TRIP_PRICES_API_URL,
    "choose_seat": settings.CHOOSE_SEAT_API_URL,
    "confirm_flight_change": settings.CONFIRM_FLIGHT_CHANGE_API_URL,
    "escalate_to_human_agent": settings.ESCALATE_TO_HUMAN_AGENT_API_URL,
    "get_airline_checkin_baggage_info": settings.GET_AIRLINE_CHECKIN_BAGGAGE_INFO_API_URL,
    "get_boarding_pass": settings.GET_BOARDING_PASS_API_URL,
    "get_boarding_pass_pdf": settings.GET_BOARDING_PASS_PDF_API_URL,
    "get_booking_details": settings.GET_BOOKING_DETAILS_API_URL,
    "get_check_in_info": settings.GET_CHECK_IN_INFO_API_URL,
    "get_excursion_cancellation_policy": settings.GET_EXCURSION_CANCELLATION_POLICY_API_URL,
    "get_flight_status": settings.GET_FLIGHT_STATUS_API_URL,
    "get_phone_checkin_info": settings.GET_PHONE_CHECKIN_INFO_API_URL,
    "get_refund": settings.GET_REFUND_API_URL,
    "get_trip_cancellation_policy": settings.GET_TRIP_CANCELLATION_POLICY_API_URL,
    "get_trip_segments": settings.GET_TRIP_SEGMENTS_API_URL,
    "initiate_refund": settings.INITIATE_REFUND_API_URL,
    "issue_travel_credit_voucher": settings.ISSUE_TRAVEL_CREDIT_VOUCHER_API_URL,
    "issue_travel_voucher": settings.ISSUE_TRAVEL_VOUCHER_API_URL,
    "purchase_flight_insurance": settings.PURCHASE_FLIGHT_INSURANCE_API_URL,
    "purchase_trip_insurance": settings.PURCHASE_TRIP_INSURANCE_API_URL,
    "query_airport_checkin_info": settings.QUERY_AIRPORT_CHECKIN_INFO_API_URL,
    "query_booking_details": settings.QUERY_BOOKING_DETAILS_API_URL,
    "query_compensation_eligibility": settings.QUERY_COMPENSATION_ELIGIBILITY_API_URL,
    "query_flight_availability": settings.QUERY_FLIGHT_AVAILABILITY_API_URL,
    "resend_boarding_pass": settings.RESEND_BOARDING_PASS_API_URL,
    "retrieve_booking_by_email": settings.RETRIEVE_BOOKING_BY_EMAIL_API_URL,
    "retrieve_flight_insurance": settings.RETRIEVE_FLIGHT_INSURANCE_API_URL,
    "schedule_callback": settings.SCHEDULE_CALLBACK_API_URL,
    "search_flight": settings.SEARCH_FLIGHT_API_URL,
    "search_flight_prices": settings.SEARCH_FLIGHT_PRICES_API_URL,
    "search_flights": settings.SEARCH_FLIGHTS_API_URL,
    "search_trip": settings.SEARCH_TRIP_API_URL,
    "search_trip_prices": settings.SEARCH_TRIP_PRICES_API_URL,
    "send_boarding_pass_email": settings.SEND_BOARDING_PASS_EMAIL_API_URL,
    "send_email": settings.SEND_EMAIL_API_URL,
    "send_itinerary_email": settings.SEND_ITINERARY_EMAIL_API_URL,
    "update_flight_date": settings.UPDATE_FLIGHT_DATE_API_URL,
    "update_refund_method": settings.UPDATE_REFUND_METHOD_API_URL,
    "verify_booking_and_get_boarding_pass": settings.VERIFY_BOOKING_AND_GET_BOARDING_PASS_API_URL,
}

def _args_to_query_string(args: dict) -> str:
    """
    Convert tool_call 'arguments' into a readable query string for RAG API.
    Example: {'origin': 'JFK', 'destination': 'LHR'} => "origin=JFK, destination=LHR"
    """
    return ", ".join([f"{k}={v}" for k, v in args.items()])

def call_travel_or_rag_api(state: dict) -> dict:
    if not state.get("tool_call"):
        return {**state, "tool_output": "No tool_call found."}

    try:
        tool = json.loads(state["tool_call"])
        name = tool.get("name")
        args = tool.get("arguments", {})

        # RAG tool calls
        if name in RAG_INTENTS:
            if name == "query_policy_rag_db":
                query = args["query"]  # Required field
            else:
                query = _args_to_query_string(args)

            r = httpx.post(RAG_API_URL, json={"query": query, "max_results": TOP_K_RESULTS})
            return {**state, "tool_output": r.json().get("answer", r.text)}

        # Normal POST APIs
        elif name in INTENT_ROUTING_MAP:
            endpoint_path = INTENT_ROUTING_MAP[name]
            full_url = urljoin(settings.NON_AI_API_URL, endpoint_path)
            r = httpx.post(full_url, json=args)
            return {**state, "tool_output": r.json().get("data", r.text)}

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
