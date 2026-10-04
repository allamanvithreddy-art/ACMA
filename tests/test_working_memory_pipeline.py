import pytest
import time
from datetime import datetime, timezone, timedelta
from memory.schema import Memory
from memory.store import MemoryStore
from memory.working_memory import WorkingMemory, WorkingMemoryItem
from conflict.pipeline import ACPipeline


def test_working_memory_user_and_session_isolation(tmp_path):
    wm = WorkingMemory(default_capacity=10, default_ttl_seconds=300)

    # User 1, Session A
    wm.add_memory(
        {"subject": "user1", "attribute": "project", "value": "User 1 working on ACMA"},
        user_id="user_1",
        session_id="session_A",
    )
    wm.set_task_context("current_task", "ACMA Development", user_id="user_1", session_id="session_A")

    # User 2, Session A
    wm.add_memory(
        {"subject": "user2", "attribute": "project", "value": "User 2 working on Robotics"},
        user_id="user_2",
        session_id="session_A",
    )
    wm.set_task_context("current_task", "Robotics Navigation", user_id="user_2", session_id="session_A")

    # User 1, Session B
    wm.add_memory(
        {"subject": "user1", "attribute": "hobby", "value": "User 1 playing chess"},
        user_id="user_1",
        session_id="session_B",
    )

    # Verify User 1 Session A
    u1_sa = wm.get_recent_memories(user_id="user_1", session_id="session_A")
    assert len(u1_sa) == 1
    assert "ACMA" in u1_sa[0].value
    assert wm.get_task_context("current_task", user_id="user_1", session_id="session_A") == "ACMA Development"

    # Verify User 2 Session A has zero leakage from User 1
    u2_sa = wm.get_recent_memories(user_id="user_2", session_id="session_A")
    assert len(u2_sa) == 1
    assert "Robotics" in u2_sa[0].value
    assert wm.get_task_context("current_task", user_id="user_2", session_id="session_A") == "Robotics Navigation"

    # Verify User 1 Session B has zero leakage from Session A
    u1_sb = wm.get_recent_memories(user_id="user_1", session_id="session_B")
    assert len(u1_sb) == 1
    assert "chess" in u1_sb[0].value
    assert wm.get_task_context("current_task", user_id="user_1", session_id="session_B") is None


def test_working_memory_ttl_expiration():
    wm = WorkingMemory(default_capacity=10, default_ttl_seconds=0.1)

    # Add item with 0.1s TTL
    wm.add_memory(
        {"subject": "user", "attribute": "temp", "value": "Temporary scratchpad data"},
        user_id="user_test",
        session_id="session_temp",
        ttl_seconds=0.05,
    )

    # Immediately available
    recent = wm.get_recent_memories(user_id="user_test", session_id="session_temp")
    assert len(recent) == 1

    # Wait for expiration
    time.sleep(0.1)

    # Now expired and pruned
    recent_after = wm.get_recent_memories(user_id="user_test", session_id="session_temp")
    assert len(recent_after) == 0


def test_working_memory_turn_based_expiration():
    wm = WorkingMemory(default_capacity=10, default_ttl_seconds=3600, default_max_turns=3)

    wm.add_memory(
        {"subject": "user", "attribute": "topic", "value": "Initial conversation turn topic"},
        user_id="user_turns",
        session_id="session_turns",
    )

    # Turn 1, 2, 3
    for _ in range(3):
        wm.step_turn(user_id="user_turns", session_id="session_turns")
        assert len(wm.get_recent_memories(user_id="user_turns", session_id="session_turns")) == 1

    # Turn 4 (exceeds max_turns=3)
    wm.step_turn(user_id="user_turns", session_id="session_turns")
    assert len(wm.get_recent_memories(user_id="user_turns", session_id="session_turns")) == 0


def test_pipeline_end_to_end_real_time_flow(tmp_path):
    db_file = tmp_path / "test_flow.db"
    store = MemoryStore(file_path=db_file)
    wm = WorkingMemory()
    pipeline = ACPipeline(working_memory=wm)

    # 1. Event 1: Initial preference (I prefer Java for backend development)
    ev1 = {
        "subject": "user",
        "attribute": "backend_language",
        "value": "I prefer Java for backend development",
        "scope": "general",
        "context": "backend",
    }
    res1 = pipeline.process_event(ev1, store=store, user_id="u1", session_id="s1")
    assert res1["event_decision"]["action"] == "Preserve"
    assert len(store.get_active_memories()) == 1

    # 2. Event 2: Duplicate preference (I prefer Java for backend development)
    ev2 = {
        "subject": "user",
        "attribute": "backend_language",
        "value": "I prefer Java for backend development",
        "scope": "general",
        "context": "backend",
    }
    res2 = pipeline.process_event(ev2, store=store, user_id="u1", session_id="s1")
    assert res2["event_decision"]["action"] == "Ignore"
    # Duplicate was ignored, active count remains 1
    assert len(store.get_active_memories()) == 1

    # 3. Event 3: Independent fact (I prefer dark mode)
    ev3 = {
        "subject": "user",
        "attribute": "theme",
        "value": "I prefer dark mode",
        "scope": "general",
        "context": "interface",
    }
    res3 = pipeline.process_event(ev3, store=store, user_id="u1", session_id="s1")
    assert res3["event_decision"]["action"] == "Preserve"
    assert len(store.get_active_memories()) == 2

    # 4. Event 4: Valid update (I now switched backend development to Python)
    ev4 = {
        "subject": "user",
        "attribute": "backend_language",
        "value": "I now switched backend development to Python",
        "scope": "general",
        "context": "backend",
    }
    res4 = pipeline.process_event(ev4, store=store, user_id="u1", session_id="s1")
    assert res4["event_decision"]["action"] == "Resolve"
    # Old Java memory should be superseded, new Python memory active
    active = store.get_active_memories()
    active_texts = [m.value for m in active]
    assert any("Python" in t for t in active_texts)
    assert not any("Java" in t for t in active_texts)

    all_memories = store.load_memories()
    superseded = [m for m in all_memories if m.status == "superseded"]
    assert len(superseded) == 1
    assert "Java" in superseded[0].value
