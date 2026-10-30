import json
import re
import time
from typing import Any, Callable, Dict, Optional
import httpx
from config.settings import AgentConfig


class LLMError(Exception):
    """Raised when an LLM request fails or returns unrecoverable malformed output."""


class BaseLLMClient:
    """Abstract interface for DecideIt LLM clients."""

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


def parse_json_safely(raw_text: str) -> Dict[str, Any]:
    """Extract and validate a JSON object from raw LLM text or markdown code blocks."""
    if not raw_text or not raw_text.strip():
        raise LLMError("Empty response received from LLM.")

    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            cleaned = cleaned[brace_start : brace_end + 1]

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Malformed JSON response from LLM: {exc}") from exc

    if not isinstance(parsed, dict):
        raise LLMError("Expected top-level JSON object (dict) from LLM.")
    return parsed


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini REST API client with timeouts and bounded retries."""

    def __init__(
        self,
        config: AgentConfig,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.config = config
        self._http_client = http_client

    def build_request_payload(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        self.config.validate_provider_credentials()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.gemini_model}:generateContent"
        )
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.config.gemini_api_key or "",
        }
        payload = self.build_request_payload(system_prompt, user_prompt)

        last_error: Optional[Exception] = None
        for attempt in range(self.config.llm_max_retries + 1):
            try:
                if self._http_client is not None:
                    response = self._http_client.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.config.llm_timeout_seconds,
                    )
                else:
                    with httpx.Client(timeout=self.config.llm_timeout_seconds) as client:
                        response = client.post(url, headers=headers, json=payload)

                if response.status_code in {429, 500, 502, 503, 504}:
                    raise LLMError(f"Transient HTTP {response.status_code} from Gemini API.")
                response.raise_for_status()

                body = response.json()
                candidates = body.get("candidates", [])
                if not candidates:
                    raise LLMError("Gemini response contained no candidates.")
                parts = candidates[0].get("content", {}).get("parts", [])
                raw_text = "".join(p.get("text", "") for p in parts)
                return parse_json_safely(raw_text)

            except (httpx.HTTPError, LLMError) as exc:
                last_error = exc
                if attempt < self.config.llm_max_retries:
                    time.sleep(min(2 ** attempt * 0.1, 2.0))
                else:
                    break

        raise LLMError(f"Gemini LLM request failed after retries: {last_error}")


class GroqLLMClient(BaseLLMClient):
    """Groq OpenAI-compatible REST API client with timeouts and bounded retries."""

    def __init__(
        self,
        config: AgentConfig,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.config = config
        self._http_client = http_client

    def build_request_payload(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return {
            "model": self.config.groq_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        self.config.validate_provider_credentials()
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.groq_api_key}",
        }
        payload = self.build_request_payload(system_prompt, user_prompt)

        last_error: Optional[Exception] = None
        for attempt in range(self.config.llm_max_retries + 1):
            try:
                if self._http_client is not None:
                    response = self._http_client.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.config.llm_timeout_seconds,
                    )
                else:
                    with httpx.Client(timeout=self.config.llm_timeout_seconds) as client:
                        response = client.post(url, headers=headers, json=payload)

                if response.status_code in {429, 500, 502, 503, 504}:
                    raise LLMError(f"Transient HTTP {response.status_code} from Groq API.")
                response.raise_for_status()

                body = response.json()
                choices = body.get("choices", [])
                if not choices:
                    raise LLMError("Groq response contained no choices.")
                raw_text = choices[0].get("message", {}).get("content", "")
                return parse_json_safely(raw_text)

            except (httpx.HTTPError, LLMError) as exc:
                last_error = exc
                if attempt < self.config.llm_max_retries:
                    time.sleep(min(2 ** attempt * 0.1, 2.0))
                else:
                    break

        raise LLMError(f"Groq LLM request failed after retries: {last_error}")


class MockLLMClient(BaseLLMClient):
    """Deterministic offline LLM client for testing without external API credentials."""

    def __init__(
        self,
        custom_responder: Optional[Callable[[str, str], str]] = None,
    ) -> None:
        self.custom_responder = custom_responder

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if self.custom_responder is not None:
            raw = self.custom_responder(system_prompt, user_prompt)
            return parse_json_safely(raw)

        has_solution = '"proposed_solution": null' not in user_prompt
        return {
            "root_problem_analysis": "Analysis based on submitted problem description (retrieved_information).",
            "solution_summary": (
                "Proposed solution evaluated from user submission."
                if has_solution
                else "No proposed solution was provided; candidate approaches generated below."
            ),
            "does_solution_address_problem": (
                "Partially addresses the core problem; empirical baseline verification required (estimate)."
                if has_solution
                else "Not applicable because no solution was submitted (unknown)."
            ),
            "strengths": (
                ["Targets stated problem domain directly (recommendation)."]
                if has_solution
                else []
            ),
            "weaknesses": [
                "Quantitative baseline metrics and detailed cost breakdowns are missing (unknown)."
            ],
            "assumptions": [
                "Local stakeholders and regulatory bodies will support implementation (assumption)."
            ],
            "contradictions": [],
            "recommended_improvements": [
                "Collect quantitative baseline data prior to deployment (recommendation).",
                "Establish clear monitoring indicators for environmental and social outcomes (recommendation).",
            ],
            "alternative_solutions": [
                "Phased community-led pilot programme with low-cost monitoring sensors (recommendation).",
                "Nature-based or circular resource management intervention (recommendation).",
            ],
            "uncertainty_and_limitations": [
                "Assessment relies on user-submitted descriptions without independent field measurements."
            ],
            "recommended_next_steps": [
                "Define baseline metrics, budget breakdown, and implementation timeline.",
                "Verify regulatory requirements and apply to matching funding schemes.",
            ],
        }


def get_llm_client(
    config: AgentConfig,
    http_client: Optional[httpx.Client] = None,
    custom_mock_responder: Optional[Callable[[str, str], str]] = None,
) -> BaseLLMClient:
    """Factory that returns the configured LLM client."""
    if config.llm_provider == "gemini":
        return GeminiLLMClient(config=config, http_client=http_client)
    if config.llm_provider == "groq":
        return GroqLLMClient(config=config, http_client=http_client)
    return MockLLMClient(custom_responder=custom_mock_responder)