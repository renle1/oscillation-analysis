import json
import sys
from pathlib import Path

KEYS = [
    "event",
    "channel",
    "start_t",
    "end_t",
    "duration_sec",
    "transition_reason",
    "end_reason",
]


def normalize(events):
    out = []
    for ev in events:
        row = {k: ev.get(k) for k in KEYS}
        out.append(row)
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python compare_events.py <baseline_json> <candidate_json>")
        return 1

    baseline_path = Path(sys.argv[1])
    candidate_path = Path(sys.argv[2])

    with baseline_path.open("r", encoding="utf-8") as f:
        baseline = normalize(json.load(f))
    with candidate_path.open("r", encoding="utf-8") as f:
        candidate = normalize(json.load(f))

    if baseline == candidate:
        print("PASS: exact match")
        return 0

    print("FAIL: mismatch detected")
    print("baseline length:", len(baseline))
    print("candidate length:", len(candidate))

    n = min(len(baseline), len(candidate))
    for i in range(n):
        if baseline[i] != candidate[i]:
            print(f"\nfirst mismatch at index {i}")
            print("baseline:", baseline[i])
            print("candidate:", candidate[i])
            return 2

    if len(baseline) != len(candidate):
        print("\nlength mismatch only")
        return 3

    return 4


if __name__ == "__main__":
    raise SystemExit(main())
