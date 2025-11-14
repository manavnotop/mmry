from typing import Dict, List, Union

import requests

from mmry.base.llm_base import SummarizerBase


class OpenRouterSummarizer(SummarizerBase):
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
                "max_tokens": 256,
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

        return self.generate(prompt)

    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format a conversation list into a readable string."""
        formatted_lines = []
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_lines.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted_lines)
