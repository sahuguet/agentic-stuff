#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "anthropic>=0.45.0",
# ]
# ///
# adapted from https://sketch.dev/blog/agent_loop.py

import os
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union

import anthropic

SYSTEM_PROMPT = """You are a helpful AI assistant.
        Your job is to ask 3 questions from the Proust Questionnaire to the user.
        Don't tell the user the purpose of the conversation.
        Don't stop until you have asked all 3 questions and received 3 answers from the user.
        Try to be as concise as possible.
        Feel free to include some casual chit-chat in between the questions.
        It is ok for the user to be off-topic. But bring them back to the task at hand.
        When you are done, output the answers in a JSON format, followed by the string "<END>".
        """

def main():
    try:
        print("Type 'exit' to end the conversation.\n")
        loop(LLM("claude-3-7-sonnet-latest"))
    except KeyboardInterrupt:
        print("\n\nExiting. Goodbye!")
    except Exception as e:
        print(f"\n\nAn error occurred: {str(e)}")

def loop(llm):
    msg = [{"type": "text", "text": "ready when you are."}]
    while True:
        output, tool_calls = llm(msg)
        print("Agent: ", output)
        if output.endswith("<END>"):
            raise SystemExit(0)
        msg = user_input()

def user_input():
    x = input("You: ")
    if x.lower() in ["exit", "quit"]:
        raise SystemExit(0)
    return [{"type": "text", "text": x}]

class LLM:
    def __init__(self, model):
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise ValueError("ANTHROPIC_API_KEY environment variable not found.")
        self.client = anthropic.Anthropic()
        self.model = model
        self.messages = []
        self.system_prompt = SYSTEM_PROMPT
        self.tools = []

    def __call__(self, content):
        self.messages.append({"role": "user", "content": content})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=20_000,
            system=self.system_prompt,
            messages=self.messages,
            tools=self.tools
        )
        assistant_response = {"role": "assistant", "content": []}
        tool_calls = []
        output_text = ""

        for content in response.content:
            if content.type == "text":
                text_content = content.text
                output_text += text_content
                assistant_response["content"].append({"type": "text", "text": text_content})
            else:
                raise Exception(f"Unsupported content type: {content.type}")

        self.messages.append(assistant_response)
        return output_text, tool_calls


if __name__ == "__main__":
    main()

    