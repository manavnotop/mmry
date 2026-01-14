from mmry.base.llm_base import MergerBase
from mmry.llms.openrouter_base import OpenRouterLLMBase


class OpenRouterMerger(OpenRouterLLMBase, MergerBase):
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-safeguard-20b",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout: int = 30,
    ):
        super().__init__(
            api_key, model, base_url, timeout, max_tokens=128, temperature=0.2
        )

    def merge_memories(self, old_memory: str, new_memory: str) -> str:
        """
        Merge two memory statements into a single, updated factual statement.

        Example:
            old = "User lives in Mumbai."
            new = "User works at Google in Mumbai."
            → "User lives in Mumbai and works at Google."
        """
        prompt = (
            "You are a factual knowledge merger for an AI memory system.\n"
            "Combine the two given memory statements into one concise, factual statement.\n"
            "Keep all valid facts, remove contradictions, and be precise.\n\n"
            f"Old memory: {old_memory}\n"
            f"New memory: {new_memory}\n\n"
            "Merged memory:"
        )

        return self._call_api(prompt)

    async def merge_memories_async(self, old_memory: str, new_memory: str) -> str:
        """Async version of merge_memories."""
        prompt = (
            "You are a factual knowledge merger for an AI memory system.\n"
            "Combine the two given memory statements into one concise, factual statement.\n"
            "Keep all valid facts, remove contradictions, and be precise.\n\n"
            f"Old memory: {old_memory}\n"
            f"New memory: {new_memory}\n\n"
            "Merged memory:"
        )
        return await self._call_api_async(prompt)
