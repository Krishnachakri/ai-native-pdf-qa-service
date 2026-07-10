from abc import ABC, abstractmethod
from pydantic import BaseModel
import openai
from app.core.config import settings

class BaseLLMProvider(ABC):
    """
    Abstract Base Class for LLM providers.
    Ensures that any backend (OpenAI, local models, etc.) supports structured output.
    """
    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: str, response_schema: type[BaseModel]) -> BaseModel:
        """
        Generates a structured response validated against a Pydantic schema.
        
        Args:
            prompt: The user instruction or text content.
            system_prompt: System-level constraints and formatting rules.
            response_schema: The Pydantic model class to validate the structured output.
            
        Returns:
            BaseModel: An instance of the response_schema.
        """
        pass

class OpenAILLMProvider(BaseLLMProvider):
    """
    OpenAI implementation of BaseLLMProvider.
    Uses OpenAI gpt-4o-mini and forces structured tool-calling.
    """
    def __init__(self):
        self._client = None
        self.model = "gpt-4o-mini"

    @property
    def client(self):
        if self._client is None:
            # Fall back to openai's default env detection if settings key is not explicitly provided
            api_key = settings.OPENAI_API_KEY or None
            self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def generate_response(self, prompt: str, system_prompt: str, response_schema: type[BaseModel]) -> BaseModel:
        # Extract JSON schema from the Pydantic model (Pydantic v2 compatible)
        schema = response_schema.model_json_schema()
        
        # Define the forced tool schema matching the response schema
        tool_name = "submit_qa_response"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Submits the structured answer containing citations from the context.",
                    "parameters": schema
                }
            }
        ]
        
        # Call OpenAI Chat Completions forcing the specific tool invocation
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": tool_name}}
        )
        
        message = response.choices[0].message
        if not message.tool_calls:
            raise ValueError("OpenAI API response did not include a structured tool call.")
            
        # Parse the JSON string from tool arguments back into the Pydantic model
        arguments_json = message.tool_calls[0].function.arguments
        return response_schema.model_validate_json(arguments_json)
