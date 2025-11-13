from typing import Dict, List, Union

import requests

from mmry.base.llm_base import LLMBase


class OpenRouterSummarizer(LLMBase):
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-safeguard-20b"):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def summarize(self, text: Union[str, List[Dict[str, str]]]) -> str:
        """
        Summarize text or conversation into a factual memory statement.

        Args:
            text: Either a string or a list of conversation dicts with 'role' and 'content' keys.

        Returns:
            A summarized memory statement.
        """
        if isinstance(text, str):
            # Handle plain text
            prompt = (
                "Summarize the following text into one factual memory statement "
                "about the user. Be concise and neutral.\n\n"
                f"Text: {text}\n\nMemory:"
            )
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(text, list):
            # Handle conversation format
            # Format conversation for the prompt
            conversation_text = self._format_conversation(text)
            prompt = (
                "Analyze the following conversation and extract key factual memories "
                "about the user. Summarize into one or more concise memory statements. "
                "Be neutral and factual.\n\n"
                f"Conversation:\n{conversation_text}\n\n"
                "Extracted Memories:"
            )
            messages = [{"role": "user", "content": prompt}]
        else:
            raise ValueError(
                f"text must be either str or List[Dict[str, str]], got {type(text)}"
            )

        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 256,  # Increased for conversations
            "temperature": 0.2,
        }

        resp = requests.post(self.url, headers=headers, json=data, timeout=30)
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return content

    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format a conversation list into a readable string."""
        formatted_lines = []
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_lines.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted_lines)
