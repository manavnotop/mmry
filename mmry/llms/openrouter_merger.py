import requests

from mmry.base.llm_base import LLMBase


class OpenRouterMerger(LLMBase):
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-safeguard-20b"):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def summarize(self, text: str) -> str:
        """Dummy required by base class (not used here)."""
        return text

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

        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0.2,
        }

        resp = requests.post(self.url, headers=headers, json=data, timeout=30)
        return resp.json()["choices"][0]["message"]["content"].strip()
