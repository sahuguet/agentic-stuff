import json
import os
from typing import Dict, Any
from client import PureOpenAIClient
from models import Message, Tool

def get_weather(location: str, unit: str = "celsius") -> str:
    """Mock function to simulate getting weather data."""
    # In a real application, this would call a weather API
    weather_data = {
        "San Francisco": {"celsius": 18, "fahrenheit": 64},
        "New York": {"celsius": 22, "fahrenheit": 72},
        "London": {"celsius": 15, "fahrenheit": 59},
    }
    
    if location not in weather_data:
        return f"Weather data not available for {location}"
    
    temp = weather_data[location][unit]
    unit_symbol = "°C" if unit == "celsius" else "°F"
    return f"The current temperature in {location} is {temp}{unit_symbol}"

def print_response_details(response):
    """Helper function to print detailed response information."""
    print("\nResponse Details:")
    print(f"ID: {response.id}")
    print(f"Model: {response.model}")
    print(f"Created: {response.created}")
    print(f"Service Tier: {response.service_tier}")
    print(f"System Fingerprint: {response.system_fingerprint}")
    
    print("\nToken Usage:")
    print(f"Prompt Tokens: {response.usage.prompt_tokens}")
    print(f"Completion Tokens: {response.usage.completion_tokens}")
    print(f"Total Tokens: {response.usage.total_tokens}")
    
    if response.usage.prompt_tokens_details:
        print("\nPrompt Token Details:")
        print(f"Cached Tokens: {response.usage.prompt_tokens_details.cached_tokens}")
        print(f"Audio Tokens: {response.usage.prompt_tokens_details.audio_tokens}")
    
    if response.usage.completion_tokens_details:
        print("\nCompletion Token Details:")
        print(f"Reasoning Tokens: {response.usage.completion_tokens_details.reasoning_tokens}")
        print(f"Audio Tokens: {response.usage.completion_tokens_details.audio_tokens}")
        print(f"Accepted Prediction Tokens: {response.usage.completion_tokens_details.accepted_prediction_tokens}")
        print(f"Rejected Prediction Tokens: {response.usage.completion_tokens_details.rejected_prediction_tokens}")
    
    print("\nMessage:")
    message = response.choices[0].message
    print(f"Role: {message.role}")
    print(f"Content: {message.content}")
    
    if message.tool_calls:
        print("\nTool Calls:")
        for tool_call in message.tool_calls:
            print(f"ID: {tool_call.id}")
            print(f"Type: {tool_call.type}")
            print(f"Function Name: {tool_call.function.name}")
            print(f"Function Arguments: {tool_call.function.arguments}")
    
    if message.refusal:
        print(f"Refusal: {message.refusal}")
    if message.annotations:
        print(f"Annotations: {message.annotations}")
    print(f"Finish Reason: {response.choices[0].finish_reason}")

def main():
    # Initialize the client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    client = PureOpenAIClient(api_key=api_key)
    
    # Example 1: Basic chat completion with detailed response
    print("\n=== Example 1: Basic Chat Completion with Detailed Response ===")
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What are three interesting facts about Python?")
    ]
    
    response = client.chat_completions(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )
    
    print_response_details(response)
    
    # Example 2: Function calling with tools
    print("\n=== Example 2: Function Calling with Tools ===")
    
    # Create a weather tool
    weather_tool = client.create_function_tool(
        name="get_weather",
        description="Get the current weather in a given location",
        parameters={
            "location": {
                "type": "string",
                "description": "The city name, e.g. San Francisco"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The unit of temperature"
            }
        },
        required=["location"]
    )
    
    # Create messages for weather query
    messages = [
        Message(role="system", content="You are a helpful assistant that can check the weather."),
        Message(role="user", content="What's the weather like in San Francisco?")
    ]
    
    # Make the initial request
    response = client.chat_completions(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=[weather_tool],
        tool_choice="auto"
    )
    
    print_response_details(response)
    
    # Handle the tool call
    message = response.choices[0].message
    if message.tool_calls:
        for tool_call in message.tool_calls:
            # Parse the function arguments
            function_args = json.loads(tool_call.function.arguments)
            location = function_args.get("location")
            unit = function_args.get("unit", "celsius")
            
            # Call the weather function
            weather_result = get_weather(location, unit)
            
            # Add the tool response to messages
            messages.append(message)
            messages.append(Message(
                role="tool",
                content=weather_result,
                tool_call_id=tool_call.id
            ))
            
            # Get the final response
            final_response = client.chat_completions(
                model="gpt-3.5-turbo",
                messages=messages
            )
            print("\nFinal Response:")
            print_response_details(final_response)

if __name__ == "__main__":
    main() 