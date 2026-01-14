from typing import Dict, List, Union

from mmry.base.llm_base import SummarizerBase
from mmry.llms.openrouter_base import OpenRouterLLMBase


class OpenRouterSummarizer(OpenRouterLLMBase, SummarizerBase):
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-safeguard-20b",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout: int = 30,
    ):
        super().__init__(
            api_key, model, base_url, timeout, max_tokens=256, temperature=0.2
        )

    def summarize(self, text: Union[str, List[Dict[str, str]]]) -> str:
        """
        Summarize text or conversation into a factual memory statement.

        Args:
            text: Either a string or a list of conversation dicts with 'role'
                and 'content' keys.

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
        else:
            raise ValueError(
                f"text must be either str or List[Dict[str, str]], got {type(text)}"
            )

        return self._call_api(prompt)

    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format a conversation list into a readable string."""
        formatted_lines = []
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_lines.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted_lines)

    async def summarize_async(self, text: Union[str, List[Dict[str, str]]]) -> str:
        """
        Async version of summarize.

        Args:
            text: Either a string or a list of conversation dicts with 'role'
                and 'content' keys.

        Returns:
            A summarized memory statement.
        """
        if isinstance(text, str):
            prompt = (
                "Summarize the following text into one factual memory statement "
                "about the user. Be concise and neutral.\n\n"
                f"Text: {text}\n\nMemory:"
            )
        elif isinstance(text, list):
            conversation_text = self._format_conversation(text)
            prompt = (
                "Analyze the following conversation and extract key factual memories "
                "about the user. Summarize into one or more concise memory statements. "
                "Be neutral and factual.\n\n"
                f"Conversation:\n{conversation_text}\n\n"
                "Extracted Memories:"
            )
        else:
            raise ValueError(
                f"text must be either str or List[Dict[str, str]], got {type(text)}"
            )

        return await self._call_api_async(prompt)
