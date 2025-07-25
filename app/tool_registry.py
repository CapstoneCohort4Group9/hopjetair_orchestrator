import os
import json

def get_tool_prompt_for_intent(intent: str) -> str:
    # Get absolute path to project root (/app)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json_path = os.path.join(root_dir, "INTENT_TOOL_MAP.json")  # resolves to /app/INTENT_TOOL_MAP.json

    with open(json_path, "r", encoding="utf-8") as f:
        intent_tool_map = json.load(f)

    tools = intent_tool_map.get(intent, [])
    if not tools:
        return f"No tool configuration found for intent '{intent}'."

    tool = tools[0]

    tool_entry = (
        '{\n'
        '  "type": "function",\n'
        f'  "function": {{\n'
        f'    "name": "{tool["name"]}",\n'
        f'    "description": "{tool["description"]}",\n'
        f'    "parameters": {json.dumps(tool["parameters"], indent=6)}\n'
        '  }\n'
        '}'
    )

    tool_block = "<tools>\n" + tool_entry + "\n</tools>"

    system_prompt = (
        "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. "
        "You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.\n\n"
        "Here is the available tool:\n\n"
        f"{tool_block}\n\n"
        "Return a JSON object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n"
        "<tool_call>\n"
        '{"arguments": <args-dict>, "name": <function-name>}\n'
        "</tool_call>"
    )

    return system_prompt