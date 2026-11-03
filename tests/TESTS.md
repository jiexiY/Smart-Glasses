# Tests

From the repo root, run one module:

```
PYTHONPATH=. .venv/bin/python tests/<file>.py
```

## Offline

Safe without models or API keys. Do not load Silero, Redimnet, Parakeet, or InspireFace. Import helpers under `workers/*_utils`, not `workers.audio` or `workers.vision` (those pull in the models).

- `tests/test_worker_helpers.py` — speaker cosine match and merge; face-track update and expiry; best identity; voice and face association; intent and memory events; coordinator speech and registration; PCM, chunks, utterances, voice commands, sampled frames, tracked-face registration.
- `tests/test_database.py` — `DatabaseManager` users, face and voice embeddings, chat history. Opens a temp `identities.db` and deletes it (including `-shm` and `-wal`). Never use `./database/identities.db`.
- `tests/test_api_worker_queue.py` — `APIWorker` with fake clients: PARSE_INTENT and ANALYZE_MEMORY events, empty and unknown commands, `api_error` when the client raises.

Install `requirements-test.txt` in a Python 3.11 virtual environment for these offline suites.
On Windows set `$env:PYTHONPATH = '.'` and use `.venv\Scripts\python.exe`.

The worker helper suite includes issue #8 regressions: known faces under different names; real
bundled voice vectors under different names; conflicting and ambiguous recognition; stale/missing
frames; delayed results; pending expiry; one enrollment per continuous track; and an actual SQLite
rename that preserves the UUID, canonical name, embeddings and history. The SQLite tests collect
unused connections before deleting temporary files because Windows keeps open handles locked.

Dataset inventory, baseline failure, commands and validation limits:
[`docs/identity-registration-testing.md`](../docs/identity-registration-testing.md).

## Live OpenAI

Do not run unless asked. Each needs `OPENAI_API_KEY`.

- `tests/test_openai_tool.py` — `OpenAIClient` memory, video frames from `api/simulator_resources/LongVideo.mp4`, and `parse_intent` (self-intro and third party).
- `tests/test_openai_memory.py` — `analyze_memory` extraction and dedup against known facts.
- `tests/test_api_worker_live.py` — real `APIWorker` process; PARSE_INTENT for "Hi, my name is John."; expects `intent` or `api_error`.

## Name learning

The helper suite runs 26 labeled synthetic replay cases in both shadow and auto modes (52 runs).
The original 16 cases are in `fixtures/name_learning.json`; 10 face/voice agreement regressions are
in `fixtures/name_learning_agreement.json`. Face entries may be UUID strings (score 0.95) or explicit
objects with `user_id` and optional `score`; `vision_at` overrides the frame timestamp. The database
suite checks additive migration, concurrent event deduplication, shared aliases, rejection,
persistence, cascade deletion, and isolation of evidence across policy upgrades. The fake API
worker verifies original name-source metadata survives the intent call. This is control-flow
validation, not recognition-accuracy evaluation. See `docs/verified-name-learning.md` and
`docs/name-learning-test-first.md`.
