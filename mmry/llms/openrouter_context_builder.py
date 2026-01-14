from typing import List

from mmry.base.llm_base import ContextBuilderBase
from mmry.llms.openrouter_base import OpenRouterLLMBase


class OpenRouterContextBuilder(OpenRouterLLMBase, ContextBuilderBase):
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-safeguard-20b",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout: int = 30,
    ):
        super().__init__(
            api_key, model, base_url, timeout, max_tokens=150, temperature=0.3
        )

    def build_context(self, memories: List[str]) -> str:
        """
        Summarize multiple memory statements into a coherent, compact paragraph.

        Example:
            ["User lives in Mumbai", "User works at Google", "User likes sushi"]
            → "The user lives in Mumbai, works at Google, and likes sushi."
        """
        joined = "\n".join(f"- {m}" for m in memories)
        prompt = (
            "You are an assistant that builds context summaries for an AI agent.\n"
            "Combine the following memory statements into one concise paragraph.\n"
            "Keep it factual, coherent, and human-readable.\n\n"
            f"Memories:\n{joined}\n\nContext Summary:"
        )

        return self._call_api(prompt)

    async def build_context_async(self, memories: List[str]) -> str:
        """Async version of build_context."""
        joined = "\n".join(f"- {m}" for m in memories)
        prompt = (
            "You are an assistant that builds context summaries for an AI agent.\n"
            "Combine the following memory statements into one concise paragraph.\n"
            "Keep it factual, coherent, and human-readable.\n\n"
            f"Memories:\n{joined}\n\nContext Summary:"
        )
        return await self._call_api_async(prompt)
