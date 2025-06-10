# minimal_smolagent_pure.py

import os
import json
import http.client
from smolagents.agents import ToolCallingAgent
from smolagents.models import ChatMessage, MessageRole, get_tool_call_from_text, parse_json_if_needed

# Minimal model wrapper using raw HTTPS calls
class SimpleOpenAIModel:
    def __init__(self,
                 model_name="gpt-3.5-turbo",
                 tool_name_key: str = "name",
                tool_arguments_key: str = "arguments",):
        self.model_name = model_name
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        """Sometimes APIs do not return the tool call as a specific object, so we need to parse it."""
        message.role = MessageRole.ASSISTANT  # Overwrite role if needed
        if not message.tool_calls:
            assert message.content is not None, "Message contains no content and no tool calls"
            message.tool_calls = [
                get_tool_call_from_text(message.content, self.tool_name_key, self.tool_arguments_key)
            ]
        assert len(message.tool_calls) > 0, "No tool call was found in the model output"
        for tool_call in message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        return message

    def generate(self, messages, tools=[], **kwargs):
        conn = http.client.HTTPSConnection("api.openai.com")
        payload = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "tools": [],
            "tool_choice": "auto"
        })
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        message = json.loads(data)["choices"][0]["message"]
        if not tools and "tool_calls" in message:
            message.pop("tool_calls", None)
        print(message)
        return ChatMessage(role="assistant", content=message.get("content", ""), tool_calls=message.get("tool_calls", []))

# Create the model and agent
model = SimpleOpenAIModel()
agent = ToolCallingAgent(
    name="MinimalAgent",
    tools=[],
    model=model,
    add_base_tools=False
)

# Run the agent
response = agent.run("2+2")
print(response["content"])
