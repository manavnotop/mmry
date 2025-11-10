import requests

from mmry.base.llm_base import LLMBase


class OpenRouterContextBuilder(LLMBase):
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-safeguard-20b"):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def summarize(self, text: str) -> str:
        """Not used — required by base class."""
        return text

    def build_context(self, memories: list[str]) -> str:
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

        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.3,
        }

        resp = requests.post(self.url, headers=headers, json=data, timeout=30)
        return resp.json()["choices"][0]["message"]["content"].strip()
