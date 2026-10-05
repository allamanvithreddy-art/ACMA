from pathlib import Path
from collections import Counter
import json

ROOT = Path("data/external/memora/data/quarterly")

operation_counts = Counter()
update_actual_operations = Counter()
update_converted = Counter()

total_files = 0
bad_files = 0

update_sessions = 0
with_old_value = 0
with_memory_updates = 0
with_converted_operation = 0
with_actual_update = 0

for path in ROOT.glob("*/conversations/session_*.json"):
    total_files += 1

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        bad_files += 1
        continue

    operation = data.get("operation")
    operation_details = data.get("operation_details") or {}

    operation_counts[str(operation)] += 1

    if operation == "update":
        update_sessions += 1

        actual_operation = operation_details.get("actual_operation")
        if actual_operation is not None:
            update_actual_operations[str(actual_operation)] += 1

        if actual_operation == "update":
            with_actual_update += 1

        if operation_details.get("old_value") is not None:
            with_old_value += 1

        if isinstance(operation_details.get("memory_updates"), list) and operation_details["memory_updates"]:
            with_memory_updates += 1

        converted = operation_details.get("operation_converted")
        if converted is not None:
            with_converted_operation += 1
            update_converted[str(converted)] += 1

print("=" * 70)
print("MEMORA QUARTERLY UPDATE PROFILE")
print("=" * 70)

print(f"Total session files:              {total_files}")
print(f"Files with JSON errors:           {bad_files}")
print()

print("ALL OPERATIONS")
for key, value in operation_counts.most_common():
    print(f"  {key:25s}: {value}")

print()
print("UPDATE SESSIONS")
print(f"  Total update sessions:          {update_sessions}")
print(f"  actual_operation == update:     {with_actual_update}")
print(f"  Has old_value:                  {with_old_value}")
print(f"  Has memory_updates:             {with_memory_updates}")
print(f"  Has operation_converted:        {with_converted_operation}")

print()
print("actual_operation VALUES")
for key, value in update_actual_operations.most_common():
    print(f"  {key:25s}: {value}")

print()
print("operation_converted VALUES")
for key, value in update_converted.most_common():
    print(f"  {key:25s}: {value}")

print("=" * 70)
