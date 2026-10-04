from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


class LLMFallback:
    def __init__(
        self,
        client: OpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self.model = model or os.getenv("ACMA_LLM_MODEL")
        self.client = client

        if self.client is None and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI()

    @property
    def enabled(self) -> bool:
        return self.client is not None and bool(self.model)

    def resolve_ambiguity(
        self,
        *,
        old_memory: dict[str, Any],
        new_memory: dict[str, Any],
        pipeline_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.enabled:
            return {
                "status": "unavailable",
                "action": "Ask",
                "reason": (
                    "LLM fallback is not configured. Preserve the ambiguity "
                    "and request user clarification."
                ),
            }

        payload = {
            "old_memory": old_memory,
            "new_memory": new_memory,
            "pipeline_evidence": pipeline_evidence,
        }

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Analyze the supplied memory evidence. Do not invent "
                        "facts or external knowledge. A contradiction alone "
                        "does not prove that the newer memory permanently "
                        "replaces the older one. Return one JSON object with "
                        "keys action, reason, and unresolved_questions. "
                        "action must be one of Ignore, Preserve, Resolve, Ask. "
                        "Use Resolve only when the supplied evidence explicitly "
                        "supports replacing the old memory. If uncertain, use Ask."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
        )

        content = response.choices[0].message.content
        if not content:
            return {
                "status": "invalid_response",
                "action": "Ask",
                "reason": "The LLM returned an empty response.",
            }

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            return {
                "status": "invalid_response",
                "action": "Ask",
                "reason": "The LLM returned invalid JSON.",
            }

        allowed_actions = {"Ignore", "Preserve", "Resolve", "Ask"}
        if result.get("action") not in allowed_actions:
            return {
                "status": "invalid_response",
                "action": "Ask",
                "reason": "The LLM returned an unsupported action.",
            }

        result["status"] = "ok"
        result["model"] = self.model
        return result