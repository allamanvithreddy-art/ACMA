from pathlib import Path
from collections import Counter
import json

ROOT = Path("data/external/memora/data/quarterly")

operation_counts = Counter()
session_type_counts = Counter()
update_type_counts = Counter()
memory_update_action_counts = Counter()
converted_counts = Counter()

samples = []

for path in ROOT.glob("*/conversations/session_*.json"):
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        continue

    operation = d.get("operation")
    session_type = d.get("session_type")
    details = d.get("operation_details") or {}

    operation_counts[str(operation)] += 1
    session_type_counts[str(session_type)] += 1

    update_type = details.get("update_type")
    if update_type:
        update_type_counts[str(update_type)] += 1

    converted = details.get("operation_converted")
    if converted:
        converted_counts[str(converted)] += 1

    memory_updates = details.get("memory_updates")

    if isinstance(memory_updates, list):
        for change in memory_updates:
            if isinstance(change, dict):
                action = change.get("action")
                if action:
                    memory_update_action_counts[str(action)] += 1

                if len(samples) < 20:
                    samples.append({
                        "file": str(path),
                        "operation": operation,
                        "session_type": session_type,
                        "update_type": update_type,
                        "operation_converted": converted,
                        "change": change,
                    })

print("=" * 75)
print("MEMORA CHANGE PROFILE")
print("=" * 75)

print("\nSESSION TYPES")
for key, value in session_type_counts.most_common():
    print(f"{key:35s}: {value}")

print("\nUPDATE TYPES")
for key, value in update_type_counts.most_common():
    print(f"{key:35s}: {value}")

print("\nMEMORY UPDATE ACTIONS")
for key, value in memory_update_action_counts.most_common():
    print(f"{key:35s}: {value}")

print("\nCONVERTED OPERATIONS")
for key, value in converted_counts.most_common():
    print(f"{key:35s}: {value}")

print("\nSAMPLE MEMORY CHANGES")
for i, sample in enumerate(samples, 1):
    print(f"\n--- SAMPLE {i} ---")
    print("file:", sample["file"])
    print("operation:", sample["operation"])
    print("session_type:", sample["session_type"])
    print("update_type:", sample["update_type"])
    print("operation_converted:", sample["operation_converted"])
    print("change:", sample["change"])

print("\n" + "=" * 75)
