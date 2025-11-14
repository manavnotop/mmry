from typing import List

import requests

from mmry.base.llm_base import ContextBuilderBase


class OpenRouterContextBuilder(ContextBuilderBase):
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-safeguard-20b",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.model = model
        self.url = base_url
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """Generic method to call the LLM with a prompt"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.3,
            }

            resp = requests.post(
                self.url, headers=headers, json=data, timeout=self.timeout
            )
            resp.raise_for_status()

            response_data = resp.json()
            if "choices" not in response_data or not response_data["choices"]:
                raise ValueError("Invalid response from LLM API: no choices returned")

            content = response_data["choices"][0]["message"]["content"].strip()
            return content
        except requests.exceptions.HTTPError as e:
            raise ConnectionError(f"HTTP error during LLM call: {e}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Timeout during LLM call to {self.url}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Request error during LLM call: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Invalid response format from LLM API: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during LLM call: {str(e)}")

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

        return self.generate(prompt)
