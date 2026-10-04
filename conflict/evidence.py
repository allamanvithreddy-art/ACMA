from dataclasses import dataclass


@dataclass
class EvidenceResult:

    evidence_score: float

    nli_label: str

    nli_confidence: float

    score_margin: float

    explanation: dict