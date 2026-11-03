import os

import httpx

from app.models import GeneratedText


class ModelUnavailable(Exception):
    pass


class ModelTimeout(Exception):
    pass


class OllamaClient:
    async def generate(self, messages: list[dict[str, str]]) -> str:
        url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        timeout = float(os.getenv("OLLAMA_TIMEOUT", "180"))
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                response = await client.post(
                    f"{url}/api/chat",
                    json={
                        "model": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
                        "messages": messages,
                        "format": GeneratedText.model_json_schema(),
                        "stream": False,
                        "think": False,
                        "options": {"temperature": 0, "num_predict": 1800},
                    },
                )
                response.raise_for_status()
                data = response.json()
                content = data["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Empty model response")
                return content
        except httpx.TimeoutException as exc:
            raise ModelTimeout from exc
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ModelUnavailable from exc
