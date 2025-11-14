import requests

from mmry.base.llm_base import MergerBase


class OpenRouterMerger(MergerBase):
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
                "max_tokens": 128,
                "temperature": 0.2,
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

        return self.generate(prompt)
