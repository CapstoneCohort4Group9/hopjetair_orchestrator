# hopjetair_orchestrator

```
curl --location 'http://localhost:8005/chat' \
--header 'Content-Type: application/json' \
--data '{
    "message": "My booking reference number is YEYE2K",
    "messages": [
        {
            "role": "user",
            "content": "[intent=check_baggage_allowance][sentiment=Neutral] I'\''d like to know about my checked and carry-on baggage allowance. How could I find more information?"
        },
        {
            "role": "assistant",
            "content": "Thought: The user wants to know about their checked and carry-on baggage allowance and how to find more information. I should provide general info about baggage allowance and then ask for their booking reference to give specific details.\nAction: Provide general info and ask for booking reference.\nResponse: Sure! I can help you with that. To give you the exact baggage allowance for your flight, could you please provide your booking reference or ticket class? Also, if you want, I can guide you on how to check your baggage allowance online or through our app."
        }
    ]
}'
```
# Response

```
{
    "messages": [
        {
            "role": "user",
            "content": "[intent=check_baggage_allowance][sentiment=Neutral] I'd like to know about my checked and carry-on baggage allowance. How could I find more information?"
        },
        {
            "role": "assistant",
            "content": "Thought: The user wants to know about their checked and carry-on baggage allowance and how to find more information. I should provide general info about baggage allowance and then ask for their booking reference to give specific details.\nAction: Provide general info and ask for booking reference.\nResponse: Sure! I can help you with that. To give you the exact baggage allowance for your flight, could you please provide your booking reference or ticket class? Also, if you want, I can guide you on how to check your baggage allowance online or through our app."
        },
        {
            "role": "user",
            "content": "[intent=check_baggage_allowance][sentiment=Neutral] My booking reference number is YEYE2K"
        },
        {
            "role": "assistant",
            "content": "Thought: The user has provided a booking reference. I can now look up their specific baggage allowance.\nAction: Call check_baggage_allowance API with booking reference.\n<tool_call> {\"name\": \"check_baggage_allowance\", \"arguments\": {\"booking_reference\": \"YEYE2K\"}}</tool_call> \nResponse: Let me check the baggage allowance for your booking reference."
        },
        {
            "role": "<|tool|>",
            "content": "<tool_response>{'booking_reference': 'YEYE2K', 'status': 'Checked-in', 'location': 'ORD Terminal 2', 'expected_departure': '2025-07-08T12:00:00Z'}</tool_response>"
        }
    ]
}
```
