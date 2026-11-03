import argparse
import gc
import json
import tempfile
from collections import deque
from pathlib import Path
from typing import Any
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from database.database import DatabaseManager
from workers.api_utils.commands import intent_event
from workers.coordinator_utils.handlers import CoordinatorState, handle_event


class Commands:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def put_nowait(self, item: dict[str, Any]) -> None:
        self.items.append(item)


def replay_case(case: dict[str, Any], mode: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as folder:
        db = DatabaseManager(str(Path(folder) / "replay.db"))
        for person in case["people"]:
            db.create_user(person["id"], person["name"])
            db.save_voice_embedding(person["id"], np.array(person["voice"], dtype=np.float32))
        llm, vision, audio = Commands(), Commands(), Commands()
        state = CoordinatorState(
            db, llm, vision, audio, deque(), deque(), deque(maxlen=10), None,
            10.0, 30.0, 20.0, 640 * 480, 320, 240, 400, 0,
            name_learning_mode=mode,
        )
        with redirect_stdout(StringIO()):
            for observation in case["observations"]:
                timestamp = observation["at"]
                faces = [({"user_id": face, "track_id": index, "score": 0.95, "emb": None}
                          if isinstance(face, str) else dict(face, track_id=index, emb=None))
                         for index, face in enumerate(observation.get("faces", []))]
                state.vision_cache.append((observation.get("vision_at", timestamp), faces))
                now = observation.get("received_at", timestamp + 0.1)
                with patch("workers.coordinator_utils.handlers.time.time", return_value=now):
                    handle_event({
                        "type": "speech", "id": observation["event_id"],
                        "text": observation["text"], "time_start": timestamp,
                        "user_id": observation.get("speaker_id", "unknown"),
                        "embedding": np.array(observation["voice"], dtype=np.float32),
                    }, state)
                    event = intent_event(llm.items[-1], {
                        "cmd": "REGISTER_IDENTITY", "args": {
                            "name": observation["name"],
                            "is_self_introduction": observation.get("self_intro", True),
                        },
                    }, now)
                    handle_event(event, state)
        expected = {tuple(item) for item in case["expected_aliases"]}
        proposed = {(decision["user_id"], decision["name_key"])
                    for decision in state.name_decisions
                    if decision["status"] in {"confirmed", "would_confirm"}}
        saved = {(uid, row["name_key"]) for uid in db.get_all_users()
                 for row in db.get_identity_names(uid) if row["status"] == "confirmed"}
        added_users = len(db.get_all_users()) - len(case["people"])
        expected_statuses = ["would_confirm" if mode == "shadow" and status == "confirmed" else status
                             for status in case["expected_decisions"]]
        actual_statuses = [decision["status"] for decision in state.name_decisions]
        profiles_unchanged = (
            db.get_all_users() == {person["id"]: person["name"] for person in case["people"]}
            and db.get_all_faces() == {}
            and all(len(db.get_all_voices()[person["id"]]) == 1
                    and np.array_equal(db.get_all_voices()[person["id"]][0],
                                       np.array(person["voice"], dtype=np.float32))
                    for person in case["people"])
        )
        passed = (proposed == expected and saved == (expected if mode == "auto" else set())
                  and added_users == 0 and actual_statuses == expected_statuses
                  and profiles_unchanged and not vision.items and not audio.items)
        result = {
            "case": case["id"], "passed": passed,
            "correct_aliases": len(proposed & expected),
            "wrong_aliases": len(proposed - expected), "missed_aliases": len(expected - proposed),
            "new_user_records": added_users, "saved_aliases": len(saved),
            "observations": len(case["observations"]),
            "profiles_unchanged": profiles_unchanged,
            "expected_decisions": expected_statuses,
            "decisions": list(state.name_decisions),
        }
        gc.collect()
        return result


def replay(cases_path: Path, mode: str) -> dict[str, Any]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    results = [replay_case(case, mode) for case in cases]
    return {
        "scope": "synthetic coordinator/event replay; no recognition models or LLM run",
        "mode": mode, "cases": len(results),
        "passed": sum(result["passed"] for result in results),
        "wrong_aliases": sum(result["wrong_aliases"] for result in results),
        "missed_aliases": sum(result["missed_aliases"] for result in results),
        "new_user_records": sum(result["new_user_records"] for result in results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[1]
                        / "tests" / "fixtures" / "name_learning.json")
    parser.add_argument("--mode", choices=("shadow", "auto"), default="shadow")
    args = parser.parse_args()
    report = replay(args.cases, args.mode)
    print(json.dumps(report, indent=2))
    if report["passed"] != report["cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
