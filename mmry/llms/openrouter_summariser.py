import requests

from mmry.base.llm_base import LLMBase


class OpenRouterSummarizer(LLMBase):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def summarize(self, text: str) -> str:
        prompt = (
            "Summarize the following text into one factual memory statement "
            "about the user. Be concise and neutral.\n\n"
            f"Text: {text}\n\nMemory:"
        )

        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 60,
            "temperature": 0.2,
        }

        resp = requests.post(self.url, headers=headers, json=data, timeout=30)
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return content
