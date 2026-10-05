from __future__ import annotations

import csv
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory.schema import Memory
from conflict.pipeline import ACPipeline
from conflict.nli_engine import NLIEngine


SEED = 20261004
CASES_PER_FAMILY = 48
TARGET_CASES = 15 * CASES_PER_FAMILY

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "realistic_500_plus"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIONS = ["Ignore", "Preserve", "Resolve", "Ask"]


@dataclass(frozen=True)
class Case:
    case_id: str
    family: str
    domain: str
    old: Memory
    new: Memory
    expected_action: str
    expected_safe_route: bool


def make_memory(
    memory_id: str,
    *,
    subject: str,
    attribute: str,
    value: str,
    scope: str = "general",
    context: str = "general",
    time_value: str | None = None,
    metadata: dict[str, Any] | None = None,
    text: str | None = None,
) -> Memory:
    return Memory(
        memory_id=memory_id,
        user_id="benchmark-user",
        text=text if text is not None else value,
        subject=subject,
        attribute=attribute,
        value=value,
        scope=scope,
        context=context,
        time=time_value,
        source="benchmark",
        confidence=0.95,
        importance=0.70,
        metadata=metadata or {},
    )


DOMAINS = [
    {
        "domain": "development",
        "subject": "user",
        "attribute": "primary_language",
        "values": [
            ("Python", "Java"),
            ("Java", "Python"),
            ("TypeScript", "Go"),
            ("C++", "Rust"),
        ],
        "contexts": ["backend", "DSA practice", "personal projects", "work projects"],
    },
    {
        "domain": "database",
        "subject": "user",
        "attribute": "database",
        "values": [
            ("PostgreSQL", "MySQL"),
            ("SQLite", "PostgreSQL"),
            ("MySQL", "MongoDB"),
            ("MongoDB", "PostgreSQL"),
        ],
        "contexts": ["the project", "the API", "the analytics service", "the prototype"],
    },
    {
        "domain": "editor",
        "subject": "user",
        "attribute": "editor",
        "values": [
            ("VS Code", "Neovim"),
            ("IntelliJ IDEA", "VS Code"),
            ("PyCharm", "VS Code"),
            ("Eclipse", "IntelliJ IDEA"),
        ],
        "contexts": ["coding", "Java work", "Python work", "daily development"],
    },
    {
        "domain": "operating_system",
        "subject": "user",
        "attribute": "operating_system",
        "values": [
            ("Windows", "Ubuntu"),
            ("Ubuntu", "Windows"),
            ("macOS", "Windows"),
            ("Windows", "macOS"),
        ],
        "contexts": ["daily use", "development", "my laptop", "work machine"],
    },
    {
        "domain": "food",
        "subject": "user",
        "attribute": "diet",
        "values": [
            ("vegetarian", "eats meat"),
            ("vegan", "eats dairy"),
            ("avoids peanuts", "eats peanuts"),
            ("does not drink coffee", "drinks coffee"),
        ],
        "contexts": ["daily diet", "food", "restaurants", "home"],
    },
    {
        "domain": "travel",
        "subject": "user",
        "attribute": "home_city",
        "values": [
            ("Hyderabad", "Bengaluru"),
            ("Chennai", "Pune"),
            ("Mumbai", "Hyderabad"),
            ("Delhi", "Chennai"),
        ],
        "contexts": ["home", "current residence", "work week", "college"],
    },
    {
        "domain": "music",
        "subject": "user",
        "attribute": "music_preference",
        "values": [
            ("rock", "classical"),
            ("jazz", "electronic"),
            ("pop", "metal"),
            ("instrumental music", "vocal-heavy music"),
        ],
        "contexts": ["music", "daily listening", "work", "study"],
    },
    {
        "domain": "study",
        "subject": "user",
        "attribute": "study_time",
        "values": [
            ("morning", "night"),
            ("evening", "afternoon"),
            ("weekends", "weekdays"),
            ("early morning", "late night"),
        ],
        "contexts": ["studying", "exam preparation", "college", "DSA practice"],
    },
    {
        "domain": "notifications",
        "subject": "user",
        "attribute": "notification_channel",
        "values": [
            ("email", "SMS"),
            ("push notifications", "email"),
            ("Slack", "email"),
            ("SMS", "push notifications"),
        ],
        "contexts": ["notifications", "alerts", "project updates", "reminders"],
    },
    {
        "domain": "theme",
        "subject": "application",
        "attribute": "theme",
        "values": [
            ("dark mode", "light mode"),
            ("light mode", "dark mode"),
            ("system theme", "dark mode"),
            ("dark mode", "system theme"),
        ],
        "contexts": ["the app", "the dashboard", "the web client", "the admin portal"],
    },
    {
        "domain": "framework",
        "subject": "project",
        "attribute": "backend_framework",
        "values": [
            ("FastAPI", "Flask"),
            ("Express", "FastAPI"),
            ("Django", "FastAPI"),
            ("FastAPI", "Django"),
        ],
        "contexts": ["the backend", "the API", "the service", "the project"],
    },
    {
        "domain": "cloud",
        "subject": "project",
        "attribute": "cloud_provider",
        "values": [
            ("AWS", "Azure"),
            ("Azure", "GCP"),
            ("GCP", "AWS"),
            ("AWS", "GCP"),
        ],
        "contexts": ["deployment", "production", "cloud hosting", "the service"],
    },
    {
        "domain": "exercise",
        "subject": "user",
        "attribute": "exercise_routine",
        "values": [
            ("running", "cycling"),
            ("gym workouts", "swimming"),
            ("yoga", "weight training"),
            ("cycling", "running"),
        ],
        "contexts": ["exercise", "fitness", "weekdays", "weekends"],
    },
    {
        "domain": "transport",
        "subject": "user",
        "attribute": "commute",
        "values": [
            ("bus", "train"),
            ("car", "metro"),
            ("metro", "bike"),
            ("bike", "car"),
        ],
        "contexts": ["commuting", "college", "work", "daily travel"],
    },
    {
        "domain": "communication",
        "subject": "user",
        "attribute": "meeting_style",
        "values": [
            ("video calls", "phone calls"),
            ("short meetings", "long meetings"),
            ("email", "chat"),
            ("in-person meetings", "video calls"),
        ],
        "contexts": ["meetings", "work", "team communication", "project discussions"],
    },
    {
        "domain": "privacy",
        "subject": "application",
        "attribute": "data_retention",
        "values": [
            ("30 days", "7 days"),
            ("7 days", "90 days"),
            ("90 days", "30 days"),
            ("30 days", "never delete"),
        ],
        "contexts": ["logs", "user data", "analytics", "application records"],
    },
]


def rotate_domain(index: int) -> dict[str, Any]:
    return DOMAINS[index % len(DOMAINS)]


def make_case(
    number: int,
    family: str,
    domain: dict[str, Any],
    old_text: str,
    new_text: str,
    *,
    old_scope: str = "general",
    new_scope: str = "general",
    old_context: str = "general",
    new_context: str = "general",
    old_time: str | None = None,
    new_time: str | None = None,
    old_metadata: dict[str, Any] | None = None,
    new_metadata: dict[str, Any] | None = None,
    expected_action: str,
    expected_safe_route: bool,
) -> Case:
    old = make_memory(
        f"old_{number:04d}",
        subject=domain["subject"],
        attribute=domain["attribute"],
        value=old_text,
        scope=old_scope,
        context=old_context,
        time_value=old_time,
        metadata=old_metadata,
    )
    new = make_memory(
        f"new_{number:04d}",
        subject=domain["subject"],
        attribute=domain["attribute"],
        value=new_text,
        scope=new_scope,
        context=new_context,
        time_value=new_time,
        metadata=new_metadata,
    )
    return Case(
        case_id=f"case_{number:04d}",
        family=family,
        domain=domain["domain"],
        old=old,
        new=new,
        expected_action=expected_action,
        expected_safe_route=expected_safe_route,
    )


def natural_duplicate_pair(domain: dict[str, Any], idx: int) -> tuple[str, str]:
    value = domain["values"][idx % len(domain["values"])][0]
    domain_name = domain["domain"]

    exact_forms = [
        f"I use {value}.",
        f"My preferred {domain_name.replace('_', ' ')} is {value}.",
        f"I normally choose {value}.",
        f"For {domain_name.replace('_', ' ')}, I use {value}.",
    ]
    text = exact_forms[idx % len(exact_forms)]
    return text, text


def semantic_duplicate_pair(domain: dict[str, Any], idx: int) -> tuple[str, str]:
    value = domain["values"][idx % len(domain["values"])][0]
    templates = [
        (f"I use {value}.", f"My main choice is {value}."),
        (f"I prefer {value}.", f"{value} is what I generally prefer."),
        (f"My usual option is {value}.", f"I normally go with {value}."),
        (f"I work with {value}.", f"{value} is the option I typically use."),
        (f"I choose {value} for this.", f"For this, {value} is my usual choice."),
    ]
    return templates[idx % len(templates)]


def compatible_pair(domain: dict[str, Any], idx: int) -> tuple[str, str, str, str]:
    value = domain["values"][idx % len(domain["values"])][0]
    contexts = domain["contexts"]
    old_context = contexts[idx % len(contexts)]
    new_context = contexts[(idx + 1) % len(contexts)]
    return (
        f"I prefer {value} for {old_context}.",
        f"I prefer {value} for {new_context}.",
        old_context,
        new_context,
    )


def independent_attribute_pair(domain: dict[str, Any], idx: int) -> tuple[str, str, str]:
    value = domain["values"][idx % len(domain["values"])][0]
    other_attributes = [
        "favorite_color",
        "weekend_activity",
        "preferred_format",
        "notification_sound",
        "favorite_subject",
        "workspace",
    ]
    other_attribute = other_attributes[idx % len(other_attributes)]
    new_value = [
        "blue",
        "hiking",
        "PDF",
        "subtle",
        "machine learning",
        "home office",
    ][idx % 6]
    return f"I use {value}.", f"My {other_attribute.replace('_', ' ')} is {new_value}.", other_attribute


def independent_subject_pair(domain: dict[str, Any], idx: int) -> tuple[str, str, str]:
    value = domain["values"][idx % len(domain["values"])][0]
    subjects = [
        ("user", "application"),
        ("user", "project"),
        ("application", "project"),
        ("project", "team"),
    ]
    old_subject, new_subject = subjects[idx % len(subjects)]
    return f"{old_subject.title()} uses {value}.", f"{new_subject.title()} uses the internal default.", new_subject


def explicit_update_pair(domain: dict[str, Any], idx: int) -> tuple[str, str]:
    old_value, new_value = domain["values"][idx % len(domain["values"])]
    context = domain["contexts"][idx % len(domain["contexts"])]
    old_texts = [
        f"I use {old_value} for {context}.",
        f"My current choice for {context} is {old_value}.",
        f"We use {old_value} for {context}.",
    ]
    new_texts = [
        f"I now use {new_value} for {context}.",
        f"I have switched to {new_value} for {context}.",
        f"We changed the {context} setup to {new_value}.",
    ]
    return old_texts[idx % len(old_texts)], new_texts[idx % len(new_texts)]


def explicit_replacement_pair(domain: dict[str, Any], idx: int) -> tuple[str, str]:
    old_value, new_value = domain["values"][idx % len(domain["values"])]
    context = domain["contexts"][idx % len(domain["contexts"])]
    old = f"The {context} currently uses {old_value}."
    new = f"The {context} is being replaced with {new_value}."
    return old, new


def clear_conflict_pair(domain: dict[str, Any], idx: int) -> tuple[str, str]:
    old_value, new_value = domain["values"][idx % len(domain["values"])]
    context = domain["contexts"][idx % len(domain["contexts"])]
    forms = [
        (f"I always use {old_value} for {context}.", f"I always use {new_value} for {context}."),
        (f"My fixed choice for {context} is {old_value}.", f"My fixed choice for {context} is {new_value}."),
        (f"The {context} setting is {old_value}.", f"The {context} setting is {new_value}."),
    ]
    return forms[idx % len(forms)]


def event_exception_pair(domain: dict[str, Any], idx: int) -> tuple[str, str, str]:
    old_value, new_value = domain["values"][idx % len(domain["values"])]
    event_contexts = [
        "a conference",
        "a family event",
        "a trip",
        "a client meeting",
        "a one-off experiment",
        "a weekend event",
        "a presentation",
        "a workshop",
    ]
    event = event_contexts[idx % len(event_contexts)]
    old = f"I generally use {old_value}."
    new = f"I used {new_value} during {event}."
    return old, new, event


def temporary_context_pair(domain: dict[str, Any], idx: int) -> tuple[str, str, str]:
    old_value, new_value = domain["values"][idx % len(domain["values"])]
    contexts = [
        "during travel",
        "while visiting a client",
        "during an exam week",
        "during a short-term project",
        "while testing a prototype",
        "during a temporary assignment",
    ]
    context = contexts[idx % len(contexts)]
    old = f"I normally use {old_value}."
    new = f"I use {new_value} {context}."
    return old, new, context


def temporal_conflict_pair(domain: dict[str, Any], idx: int) -> tuple[str, str]:
    old_value, new_value = domain["values"][idx % len(domain["values"])]
    year = 2024 + (idx % 2)
    old = f"I use {old_value}."
    new = f"I use {new_value}."
    return old, new


def context_conflict_pair(domain: dict[str, Any], idx: int) -> tuple[str, str, str, str]:
    old_value, new_value = domain["values"][idx % len(domain["values"])]
    contexts = domain["contexts"]
    old_context = contexts[idx % len(contexts)]
    new_context = contexts[(idx + 2) % len(contexts)]
    old = f"For {old_context}, I use {old_value}."
    new = f"For {new_context}, I use {new_value}."
    return old, new, old_context, new_context


def ambiguous_change_pair(domain: dict[str, Any], idx: int) -> tuple[str, str]:
    old_value, new_value = domain["values"][idx % len(domain["values"])]
    old_forms = [
        f"I tend to use {old_value}.",
        f"I usually pick {old_value}.",
        f"My preference is {old_value}.",
        f"I work mostly with {old_value}.",
    ]
    new_forms = [
        f"Lately I use {new_value}.",
        f"I've been using {new_value} more.",
        f"These days I choose {new_value}.",
        f"Recently I work more with {new_value}.",
    ]
    return old_forms[idx % len(old_forms)], new_forms[idx % len(new_forms)]


def constraint_pair(domain: dict[str, Any], idx: int) -> tuple[str, str]:
    old_value, new_value = domain["values"][idx % len(domain["values"])]
    context = domain["contexts"][idx % len(domain["contexts"])]
    old = f"The {context} must use {old_value}."
    new = f"The {context} uses {new_value}."
    return old, new


def build_dataset() -> list[Case]:
    rng = random.Random(SEED)
    cases: list[Case] = []
    n = 1

    for family_index, family in enumerate(
        [
            "exact_duplicate",
            "semantic_duplicate",
            "compatible",
            "independent_attribute",
            "independent_subject",
            "explicit_update",
            "explicit_replacement",
            "clear_conflict",
            "event_exception",
            "temporary_context_conflict",
            "temporal_conflict",
            "context_conflict",
            "ambiguous_change",
            "constraint_violation",
            "preference_shift_without_update",
        ]
    ):
        for i in range(CASES_PER_FAMILY):
            domain = rotate_domain(family_index * CASES_PER_FAMILY + i)

            if family == "exact_duplicate":
                old_text, new_text = natural_duplicate_pair(domain, i)
                cases.append(make_case(
                    n, family, domain, old_text, new_text,
                    expected_action="Ignore", expected_safe_route=True,
                ))

            elif family == "semantic_duplicate":
                old_text, new_text = semantic_duplicate_pair(domain, i)
                cases.append(make_case(
                    n, family, domain, old_text, new_text,
                    expected_action="Ignore", expected_safe_route=False,
                ))

            elif family == "compatible":
                old_text, new_text, old_ctx, new_ctx = compatible_pair(domain, i)
                cases.append(make_case(
                    n, family, domain, old_text, new_text,
                    old_context=old_ctx, new_context=new_ctx,
                    expected_action="Preserve", expected_safe_route=True,
                ))

            elif family == "independent_attribute":
                old_text, new_text, new_attr = independent_attribute_pair(domain, i)
                new_memory = make_memory(
                    f"new_{n:04d}",
                    subject=domain["subject"],
                    attribute=new_attr,
                    value=new_text,
                    scope="general",
                    context="general",
                )
                old_memory = make_memory(
                    f"old_{n:04d}",
                    subject=domain["subject"],
                    attribute=domain["attribute"],
                    value=old_text,
                    scope="general",
                    context="general",
                )
                cases.append(Case(
                    case_id=f"case_{n:04d}",
                    family=family,
                    domain=domain["domain"],
                    old=old_memory,
                    new=new_memory,
                    expected_action="Preserve",
                    expected_safe_route=True,
                ))

            elif family == "independent_subject":
                old_text, new_text, new_subject = independent_subject_pair(domain, i)
                old_memory = make_memory(
                    f"old_{n:04d}",
                    subject=domain["subject"],
                    attribute=domain["attribute"],
                    value=old_text,
                )
                new_memory = make_memory(
                    f"new_{n:04d}",
                    subject=new_subject,
                    attribute=domain["attribute"],
                    value=new_text,
                )
                cases.append(Case(
                    case_id=f"case_{n:04d}",
                    family=family,
                    domain=domain["domain"],
                    old=old_memory,
                    new=new_memory,
                    expected_action="Preserve",
                    expected_safe_route=True,
                ))

            elif family == "explicit_update":
                old_text, new_text = explicit_update_pair(domain, i)
                old_id = f"old_{n:04d}"
                cases.append(make_case(
                    n, family, domain, old_text, new_text,
                    new_metadata={
                        "is_update": True,
                        "is_replacement": True,
                        "update_of": old_id,
                        "extraction_source": "frontend_llm",
                        "extraction_confidence": 0.98,
                    },
                    expected_action="Resolve", expected_safe_route=True,
                ))

            elif family == "explicit_replacement":
                old_text, new_text = explicit_replacement_pair(domain, i)
                old_id = f"old_{n:04d}"
                cases.append(make_case(
                    n, family, domain, old_text, new_text,
                    new_metadata={
                        "is_replacement": True,
                        "supersedes": [old_id],
                        "extraction_source": "frontend_llm",
                        "extraction_confidence": 0.99,
                    },
                    expected_action="Resolve", expected_safe_route=True,
                ))

            elif family == "clear_conflict":
                old_text, new_text = clear_conflict_pair(domain, i)
                cases.append(make_case(
                    n, family, domain, old_text, new_text,
                    expected_action="Ask", expected_safe_route=False,
                ))

            elif family == "event_exception":
                old_text, new_text, event = event_exception_pair(domain, i)
                cases.append(make_case(
                    n, family, domain, old_text, new_text,
                    old_scope="general",
                    new_scope="specific_event",
                    old_context="general",
                    new_context=event,
                    new_metadata={"is_event": True, "extraction_source": "frontend_llm"},
                    expected_action="Ask", expected_safe_route=False,
                ))

            elif family == "temporary_context_conflict":
                old_text, new_text, context = temporary_context_pair(domain, i)
                cases.append(make_case(
                    n, family, domain, old_text, new_text,
                    old_scope="general",
                    new_scope="context_specific",
                    old_context="general",
                    new_context=context,
                    expected_action="Ask", expected_safe_route=False,
                ))

            elif family == "temporal_conflict":
                old_text, new_text = temporal_conflict_pair(domain, i)
                old_year = 2023 + (i % 2)
                new_year = old_year + 1
                cases.append(make_case(
                    n, family, domain, old_text, new_text,
                    old_time=f"{old_year}-06-01T10:00:00+00:00",
                    new_time=f"{new_year}-06-01T10:00:00+00:00",
                    expected_action="Ask", expected_safe_route=False,
                ))

            elif family == "context_conflict":
                old_text, new_text, old_ctx, new_ctx = context_conflict_pair(domain, i)
                cases.append(make_case(
                    n, family, domain, old_text, new_text,
                    old_scope="context_specific",
                    new_scope="context_specific",
                    old_context=old_ctx,
                    new_context=new_ctx,
                    expected_action="Ask", expected_safe_route=False,
                ))

            elif family == "ambiguous_change":
                old_text, new_text = ambiguous_change_pair(domain, i)
                cases.append(make_case(
                    n, family, domain, old_text, new_text,
                    expected_action="Ask", expected_safe_route=False,
                ))

            elif family == "constraint_violation":
                old_text, new_text = constraint_pair(domain, i)
                old_id = f"old_{n:04d}"
                old_memory = make_memory(
                    old_id,
                    subject=domain["subject"],
                    attribute=domain["attribute"],
                    value=old_text,
                    metadata={"is_constraint": True},
                )
                new_memory = make_memory(
                    f"new_{n:04d}",
                    subject=domain["subject"],
                    attribute=domain["attribute"],
                    value=new_text,
                    metadata={"violates_memory_id": old_id},
                )
                cases.append(Case(
                    case_id=f"case_{n:04d}",
                    family=family,
                    domain=domain["domain"],
                    old=old_memory,
                    new=new_memory,
                    expected_action="Ask",
                    expected_safe_route=False,
                ))

            elif family == "preference_shift_without_update":
                old_text, new_text = ambiguous_change_pair(domain, i)
                cases.append(make_case(
                    n, family, domain, old_text, new_text,
                    old_scope="general",
                    new_scope="general",
                    old_context="general",
                    new_context="general",
                    # Deliberately no is_update / replacement metadata.
                    expected_action="Ask", expected_safe_route=False,
                ))

            n += 1

    rng.shuffle(cases)
    return cases


def precision_recall_f1(
    matrix: dict[str, Counter[str]],
    action: str,
) -> dict[str, float]:
    tp = matrix[action][action]
    fp = sum(matrix[other][action] for other in ACTIONS if other != action)
    fn = sum(matrix[action][other] for other in ACTIONS if other != action)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision_percent": round(precision * 100.0, 2),
        "recall_percent": round(recall * 100.0, 2),
        "f1_percent": round(f1 * 100.0, 2),
    }


def evaluate() -> dict[str, Any]:
    dataset = build_dataset()
    assert len(dataset) == TARGET_CASES, (len(dataset), TARGET_CASES)

    print("=" * 88)
    print("ACMA REALISTIC 500+ CANDIDATE BENCHMARK")
    print("=" * 88)
    print(f"Seed: {SEED}")
    print(f"Cases: {len(dataset)}")
    print("This benchmark evaluates the live ACPipeline candidate path.")
    print("It uses realistic cross-domain language + structured metadata, not keyword-only cases.")
    print()

    pipeline = ACPipeline(nli_engine=NLIEngine.get_default())

    results: list[dict[str, Any]] = []
    final_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    family_matrix: dict[str, Counter[str]] = defaultdict(Counter)

    safe_gate_total = 0
    safe_gate_correct = 0
    safe_gate_wrong = 0
    nli_route_total = 0
    nli_route_correct = 0
    nli_route_wrong = 0
    nli_calls = 0
    errors = 0

    all_latency_ms: list[float] = []
    nli_latency_ms: list[float] = []

    action_counts = Counter()
    expected_counts = Counter()
    safe_gate_route_counts = Counter()
    nli_label_counts = Counter()

    started = time.perf_counter()

    for i, case in enumerate(dataset, start=1):
        t0 = time.perf_counter()

        try:
            result = pipeline.evaluate_candidate(
                case.old,
                case.new,
                retrieval_score=0.85,
            )

            predicted_action = str(result["action"])
            safety = result["safety"]
            route_is_safe = safety["next_stage"] == "decision"
            nli = result.get("nli")

            elapsed_ms = float(result.get("latency_ms") or ((time.perf_counter() - t0) * 1000.0))
            all_latency_ms.append(elapsed_ms)

            if nli is not None:
                nli_calls += 1
                nli_label_counts[str(nli.get("label", "unknown"))] += 1

            if route_is_safe:
                safe_gate_total += 1
                safe_gate_route_counts["SAFE -> decision"] += 1
                if predicted_action == case.expected_action:
                    safe_gate_correct += 1
                else:
                    safe_gate_wrong += 1
            else:
                nli_route_total += 1
                safe_gate_route_counts["UNSAFE -> NLI"] += 1
                if predicted_action == case.expected_action:
                    nli_route_correct += 1
                else:
                    nli_route_wrong += 1

            is_correct = predicted_action == case.expected_action

            final_matrix[case.expected_action][predicted_action] += 1
            family_matrix[case.family][predicted_action] += 1

            action_counts[predicted_action] += 1
            expected_counts[case.expected_action] += 1

            results.append({
                "case_id": case.case_id,
                "family": case.family,
                "domain": case.domain,
                "old_text": case.old.value,
                "new_text": case.new.value,
                "old_scope": case.old.scope,
                "new_scope": case.new.scope,
                "old_context": case.old.context,
                "new_context": case.new.context,
                "expected_action": case.expected_action,
                "predicted_action": predicted_action,
                "correct": is_correct,
                "expected_safe_route": case.expected_safe_route,
                "actual_safe_route": route_is_safe,
                "safety_route": safety["next_stage"],
                "relationship": safety["relationship"],
                "nli_label": nli.get("label") if nli else "",
                "nli_confidence": nli.get("confidence") if nli else "",
                "retrieval_score": result.get("retrieval_score"),
                "latency_ms": round(elapsed_ms, 2),
                "reason": result["reason"],
            })

        except Exception as exc:
            errors += 1
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            all_latency_ms.append(elapsed_ms)

            final_matrix[case.expected_action]["ERROR"] += 1
            family_matrix[case.family]["ERROR"] += 1

            results.append({
                "case_id": case.case_id,
                "family": case.family,
                "domain": case.domain,
                "old_text": case.old.value,
                "new_text": case.new.value,
                "old_scope": case.old.scope,
                "new_scope": case.new.scope,
                "old_context": case.old.context,
                "new_context": case.new.context,
                "expected_action": case.expected_action,
                "predicted_action": "ERROR",
                "correct": False,
                "expected_safe_route": case.expected_safe_route,
                "actual_safe_route": "",
                "safety_route": "",
                "relationship": "",
                "nli_label": "",
                "nli_confidence": "",
                "retrieval_score": 0.85,
                "latency_ms": round(elapsed_ms, 2),
                "reason": f"{type(exc).__name__}: {exc}",
            })

        if i % 50 == 0 or i == len(dataset):
            correct_so_far = sum(1 for row in results if row["correct"])
            print(f"Evaluated {i:>3}/{len(dataset)} | accuracy so far: {correct_so_far / i * 100:6.2f}%")

    total_seconds = time.perf_counter() - started
    total_correct = sum(1 for row in results if row["correct"])
    overall_accuracy = total_correct / len(results) if results else 0.0

    safe_expected = sum(1 for case in dataset if case.expected_safe_route)
    unsafe_expected = len(dataset) - safe_expected

    safe_gate_expected_correct = sum(
        1
        for row in results
        if row["expected_safe_route"] is True
        and row["actual_safe_route"] is True
    )
    false_safe = sum(
        1
        for row in results
        if row["expected_safe_route"] is False
        and row["actual_safe_route"] is True
    )

    nli_assisted_correct_rate = (
        nli_route_correct / nli_route_total if nli_route_total else 0.0
    )
    safe_gate_action_accuracy = (
        safe_gate_correct / safe_gate_total if safe_gate_total else 0.0
    )

    per_action = {
        action: {
            "support": sum(final_matrix[action].values()),
            "predicted": sum(final_matrix[other][action] for other in ACTIONS),
            **precision_recall_f1(final_matrix, action),
        }
        for action in ACTIONS
    }

    family_accuracy: dict[str, Any] = {}
    for family in sorted({case.family for case in dataset}):
        family_rows = [row for row in results if row["family"] == family]
        family_correct = sum(1 for row in family_rows if row["correct"])
        family_accuracy[family] = {
            "cases": len(family_rows),
            "correct": family_correct,
            "accuracy_percent": round(family_correct / len(family_rows) * 100.0, 2),
        }

    action_order = ACTIONS + ["ERROR"]
    confusion = {
        expected: {
            predicted: final_matrix[expected][predicted]
            for predicted in action_order
            if final_matrix[expected][predicted]
        }
        for expected in ACTIONS
    }

    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
        return {
            "mean_ms": round(statistics.fmean(values), 2),
            "median_ms": round(statistics.median(values), 2),
            "p95_ms": round(ordered[p95_index], 2),
            "max_ms": round(max(values), 2),
        }

    summary = {
        "benchmark": {
            "name": "ACMA Realistic 500+ Candidate Benchmark",
            "seed": SEED,
            "target_cases": TARGET_CASES,
            "cases_evaluated": len(results),
            "domains": sorted({case.domain for case in dataset}),
            "families": sorted({case.family for case in dataset}),
            "note": (
                "Controlled synthetic benchmark using realistic natural-language "
                "memory statements and structured metadata. It is not a real-world "
                "ground-truth dataset; the next phase must use an external dataset."
            ),
        },
        "overall": {
            "correct": total_correct,
            "errors": errors,
            "accuracy_percent": round(overall_accuracy * 100.0, 2),
            "runtime_seconds": round(total_seconds, 2),
        },
        "safety_gate": {
            "expected_safe_cases": safe_expected,
            "expected_unsafe_cases": unsafe_expected,
            "safe_gate_auto_decision_cases": safe_gate_total,
            "safe_gate_auto_decision_correct": safe_gate_correct,
            "safe_gate_auto_decision_wrong": safe_gate_wrong,
            "safe_gate_action_accuracy_percent": round(safe_gate_action_accuracy * 100.0, 2),
            "safe_gate_coverage_percent": round(safe_gate_total / len(results) * 100.0, 2),
            "expected_safe_routed_to_decision": safe_gate_expected_correct,
            "false_safe_count": false_safe,
            "false_safe_rate_percent": round(false_safe / len(results) * 100.0, 2),
            "route_distribution": dict(safe_gate_route_counts),
        },
        "nli": {
            "total_calls": nli_calls,
            "call_rate_percent": round(nli_calls / len(results) * 100.0, 2),
            "nli_routed_cases": nli_route_total,
            "nli_route_action_accuracy_percent": round(nli_assisted_correct_rate * 100.0, 2),
            "nli_route_correct": nli_route_correct,
            "nli_route_wrong": nli_route_wrong,
            "label_distribution": dict(nli_label_counts),
        },
        "decision_policy": {
            "final_action_accuracy_percent": round(overall_accuracy * 100.0, 2),
            "predicted_distribution": dict(action_counts),
            "expected_distribution": dict(expected_counts),
            "per_action": per_action,
            "confusion_matrix": confusion,
        },
        "by_family": family_accuracy,
        "latency_ms": stats(all_latency_ms),
        "pipeline_statistics": pipeline.statistics(),
        "outputs": {
            "csv": str(OUTPUT_DIR / "realistic_720_results.csv"),
            "summary": str(OUTPUT_DIR / "realistic_720_summary.json"),
        },
    }

    csv_path = OUTPUT_DIR / "realistic_720_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_path = OUTPUT_DIR / "realistic_720_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print()
    print("=" * 88)
    print("FINAL RESULTS")
    print("=" * 88)
    print(f"Total cases:                 {len(results)}")
    print(f"Total correct:               {total_correct}")
    print(f"Total errors:                {errors}")
    print(f"Overall accuracy:             {overall_accuracy * 100:.2f}%")
    print()
    print("SAFETY GATE")
    print(f"Safe-gate auto decisions:    {safe_gate_total}")
    print(f"Safe-gate correct:            {safe_gate_correct}")
    print(f"Safe-gate action accuracy:    {safe_gate_action_accuracy * 100:.2f}%")
    print(f"Safe-gate coverage:           {safe_gate_total / len(results) * 100:.2f}%")
    print(f"False-safe cases:             {false_safe}")
    print()
    print("NLI / DECISION POLICY")
    print(f"NLI calls:                    {nli_calls}")
    print(f"NLI route cases:              {nli_route_total}")
    print(f"NLI-route action accuracy:    {nli_assisted_correct_rate * 100:.2f}%")
    print(f"Final Decision accuracy:      {overall_accuracy * 100:.2f}%")
    print()
    print("Decision distribution:")
    for action in ACTIONS:
        print(f"  {action:8s} expected={expected_counts[action]:3d} predicted={action_counts[action]:3d}")
    print()
    print(f"CSV:     {csv_path}")
    print(f"Summary: {json_path}")

    return summary


if __name__ == "__main__":
    evaluate()
