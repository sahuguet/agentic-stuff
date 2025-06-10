import json
import time
import uuid
from typing import Optional, List, Dict, Any, Union, Callable, Literal
import requests
from models import (
    Message, ChatCompletionRequest, ChatCompletionResponse,
    Tool, FunctionDefinition, FunctionParameter
)

class PureOpenAIClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def create_function_tool(
        self,
        name: str,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        required: Optional[List[str]] = None
    ) -> Tool:
        """Helper method to create a function tool definition."""
        if parameters is None:
            parameters = {}
        if required is None:
            required = []
            
        function_param = FunctionParameter(
            type="object",
            properties=parameters,
            required=required
        )
        
        function_def = FunctionDefinition(
            name=name,
            description=description,
            parameters=function_param
        )
        
        return Tool(function=function_def)

    def chat_completions(
        self,
        model: str,
        messages: List[Message],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        stop: Optional[Union[str, List[str]]] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        user: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[Literal["auto", "none"], Dict[str, Any]]] = None
    ) -> ChatCompletionResponse:
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            tools=tools,
            tool_choice=tool_choice
        )
        
        response_data = self._make_request(
            method="POST",
            endpoint="/chat/completions",
            data=request.dict(exclude_none=True)
        )
        
        print(response_data)
        return ChatCompletionResponse(**response_data)

    def chat_completions_with_functions(
        self,
        model: str,
        messages: List[Message],
        functions: List[Dict[str, Any]],
        function_call: Optional[Union[Literal["auto", "none"], Dict[str, Any]]] = None,
        **kwargs
    ) -> ChatCompletionResponse:
        """Helper method for backward compatibility with older function calling API."""
        tools = [
            Tool(function=FunctionDefinition(**func))
            for func in functions
        ]
        
        return self.chat_completions(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=function_call,
            **kwargs
        ) 