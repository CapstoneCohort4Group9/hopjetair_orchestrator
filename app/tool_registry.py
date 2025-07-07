import json

INTENT_TOOL_MAP = {
    "search_flight": [
        {
            "name": "search_flight",
            "description": "search_flight(origin: str, destination: str, departure_date: str, trip_type: str) -> list - Search for available flights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "departure_date": {"type": "string"},
                    "trip_type": {"type": "string"},
                },
                "required": ["origin", "destination", "departure_date", "trip_type"]
            }
        }
    ],
    "book_flight": [
        {
            "name": "book_flight",
            "description": "book_flight(origin: str, destination: str, departure_date: str, trip_type: str, airline: str, price: number) -> dict - Book a selected flight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "departure_date": {"type": "string"},
                    "trip_type": {"type": "string"},
                    "airline": {"type": "string"},
                    "price": {"type": "number"},
                },
                "required": ["origin", "destination", "departure_date", "trip_type", "airline", "price"]
            }
        }
    ],
    # Add other intents here (up to 34 total)
}

def get_tool_prompt_for_intent(intent: str) -> str:
    tools = INTENT_TOOL_MAP.get(intent, [])
    tool_entries = []

    for tool in tools:
        entry = (
            '{\n'
            '  "type": "function",\n'
            f'  "function": {{\n'
            f'    "name": "{tool["name"]}",\n'
            f'    "description": "{tool["description"]}",\n'
            f'    "parameters": {json.dumps(tool["parameters"], indent=6)}\n'
            '  }\n'
            '}'
        )
        tool_entries.append(entry)

    tool_block = "<tools>\n" + ",\n".join(tool_entries) + "\n</tools>"

    system_prompt = (
        "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. "
        "You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.\n\n"
        "Here are the available tools:\n\n"
        f"{tool_block}\n\n"
        "For each function call return a JSON object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n"
        "<tool_call>\n"
        '{"arguments": <args-dict>, "name": <function-name>}\n'
        "</tool_call>"
    )
    return system_prompt
