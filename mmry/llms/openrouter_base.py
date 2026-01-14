import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mmry.errors import LLMConnectionError, LLMError, LLMTTimeoutError


class OpenRouterLLMBase:
    """Base class for OpenRouter LLM implementations with shared API logic."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-safeguard-20b",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout: int = 30,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ):
        self.api_key = api_key
        self.model = model
        self.url = base_url
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((LLMConnectionError, LLMTTimeoutError)),
    )
    def _call_api(self, prompt: str) -> str:
        """Make an API call to OpenRouter with the given prompt."""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            resp = requests.post(
                self.url, headers=headers, json=data, timeout=self.timeout
            )
            resp.raise_for_status()

            response_data = resp.json()
            if "choices" not in response_data or not response_data["choices"]:
                raise LLMError("Invalid response from LLM API: no choices returned")

            content = response_data["choices"][0]["message"]["content"].strip()
            return content
        except requests.exceptions.HTTPError as e:
            raise LLMConnectionError(f"HTTP error during LLM call: {e}")
        except requests.exceptions.Timeout:
            raise LLMTTimeoutError(f"Timeout during LLM call to {self.url}")
        except requests.exceptions.RequestException as e:
            raise LLMConnectionError(f"Request error during LLM call: {str(e)}")
        except KeyError as e:
            raise LLMError(f"Invalid response format from LLM API: {str(e)}")
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Unexpected error during LLM call: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((LLMConnectionError, LLMTTimeoutError)),
    )
    async def _call_api_async(self, prompt: str) -> str:
        """Make an async API call to OpenRouter with the given prompt."""
        try:
            import httpx

            headers = {"Authorization": f"Bearer {self.api_key}"}
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.url, headers=headers, json=data)
                resp.raise_for_status()

                response_data = resp.json()
                if "choices" not in response_data or not response_data["choices"]:
                    raise LLMError("Invalid response from LLM API: no choices returned")

                content = response_data["choices"][0]["message"]["content"].strip()
                return content
        except httpx.HTTPStatusError as e:
            raise LLMConnectionError(f"HTTP error during LLM call: {e}")
        except httpx.Timeout:
            raise LLMTTimeoutError(f"Timeout during LLM call to {self.url}")
        except httpx.RequestError as e:
            raise LLMConnectionError(f"Request error during LLM call: {str(e)}")
        except KeyError as e:
            raise LLMError(f"Invalid response format from LLM API: {str(e)}")
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Unexpected error during LLM call: {str(e)}")

    def generate(self, prompt: str) -> str:
        """Generic method to call the LLM with a prompt."""
        return self._call_api(prompt)

    async def generate_async(self, prompt: str) -> str:
        """Async method to call the LLM with a prompt."""
        return await self._call_api_async(prompt)
