from __future__ import annotations

from threading import Lock
from typing import Any, Union

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from memory.schema import Memory


class NLIEngine:
    _default_instance: NLIEngine | None = None
    _instance_lock = Lock()

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: str | None = None,
    ) -> None:

        self.model_name = model_name
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self._lock = Lock()
        self._load_lock = Lock()

        self.tokenizer = None
        self.model = None
        self.id2label: dict[int, str] = {}

    @classmethod
    def get_default(cls) -> "NLIEngine":
        with cls._instance_lock:
            if cls._default_instance is None:
                cls._default_instance = cls()
            return cls._default_instance

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return

        with self._load_lock:
            if self.model is not None:
                return

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )

            self.model = (
                AutoModelForSequenceClassification
                .from_pretrained(self.model_name)
                .to(self.device)
            )

            self.model.eval()

            raw_labels = self.model.config.id2label

            self.id2label = {
                int(index): str(label).casefold()
                for index, label in raw_labels.items()
            }

    @staticmethod
    def _text(
        memory: Union[
            Memory,
            str,
            dict[str, Any],
        ],
    ) -> str:

        if isinstance(memory, Memory):
            return str(
                memory.text
                or memory.value
                or ""
            ).strip()

        if isinstance(memory, dict):
            return str(
                memory.get("text")
                or memory.get("value")
                or ""
            ).strip()

        return str(memory or "").strip()

    def compare(
        self,
        old_memory: Union[
            Memory,
            str,
            dict[str, Any],
        ],
        new_memory: Union[
            Memory,
            str,
            dict[str, Any],
        ],
    ) -> dict[str, Any]:

        premise = self._text(old_memory)
        hypothesis = self._text(new_memory)

        if not premise or not hypothesis:
            return {
                "label": "neutral",
                "confidence": 0.5,
                "margin": 0.0,
                "scores": {
                    "entailment": 0.0,
                    "neutral": 0.5,
                    "contradiction": 0.0,
                },
                "model": self.model_name,
                "premise": premise,
                "hypothesis": hypothesis,
            }

        normalized_premise = " ".join(
            premise.casefold().split()
        )
        normalized_hypothesis = " ".join(
            hypothesis.casefold().split()
        )

        if normalized_premise == normalized_hypothesis:
            return {
                "label": "entailment",
                "confidence": 1.0,
                "margin": 1.0,
                "scores": {
                    "entailment": 1.0,
                    "neutral": 0.0,
                    "contradiction": 0.0,
                },
                "model": self.model_name,
                "premise": premise,
                "hypothesis": hypothesis,
            }

        self._ensure_loaded()

        encoded = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with self._lock, torch.inference_mode():
            logits = self.model(**encoded).logits[0]

            probabilities = (
                torch.softmax(
                    logits,
                    dim=-1,
                )
                .detach()
                .cpu()
                .tolist()
            )

        scores = {
            self.id2label[index]: float(probability)
            for index, probability in enumerate(probabilities)
        }

        label = max(
            scores,
            key=scores.get,
        )

        ranked = sorted(
            scores.values(),
            reverse=True,
        )

        margin = (
            ranked[0] - ranked[1]
            if len(ranked) > 1
            else ranked[0]
        )

        return {
            "label": label,
            "confidence": float(scores[label]),
            "margin": float(margin),
            "scores": scores,
            "model": self.model_name,
            "premise": premise,
            "hypothesis": hypothesis,
        }
