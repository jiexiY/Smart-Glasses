# Issue 8: uncertain recognition must not create a person

The existing fork at `2e44f9f032730ab0b13fa0a8a1de8c671d2a5f87` still created a UUID
after three unknown-face observations over 0.5 seconds. An existing person missed by both
recognizers could therefore become a second contact under a nickname.

## Test-first result

The first 12 tests in `tests/test_identity_scenarios.py` were run against that unchanged
implementation before editing the handler. The recognized-face and recognized-voice situations
passed. The third situation failed with `Uncertain recognition must not create another UUID`:
the temporary database contained the saved Shrish plus a newly created Shri. Recovery and
expiry checks failed for the same premature creation. Explicit-confirmation tests errored because
the old implementation had no request-bound confirmation contract; those are new behavior,
not evidence of a recognition error.

The final suite adds four boundary checks, for 16 tests in total:

| Situation | Expected behavior |
| --- | --- |
| Saved face, new nickname | Reuse the UUID; preserve the canonical name and history |
| Unclear face, unique matching saved voice | Reuse the UUID for the self-introduction |
| Both unclear, including repeated unknown observations | Stay pending; retry; never create on expiry |
| Different person with an existing display name | Create a separate UUID only after explicit new-person confirmation |

Other checks cover face/voice recovery, continuous unknown expiry, idle expiry, late/repeated
confirmations, reused track IDs, insufficient observations, missing/stale/changed/multiple targets,
conflicting evidence, final database rechecks, and attempts to confirm through LLM arguments.

## Behavior change

`REGISTER_IDENTITY` can resolve an existing person or create a pending request. It cannot create
a person by itself. Each pending request gets a unique request ID and retains its original name,
track, utterance time and voice evidence. Vision events retry recognition; idle coordinator polls
also expire requests. The five-second lifetime and one-second tracking gap remain unchanged.

New enrollment is a separate local event:

```python
{"type": "confirm_enrollment", "request_id": "exact-live-request-id", "confirmed": True}
```

The coordinator checks the current sole track, fresh frame, stable observations, current users
and current voice profiles before writing. A recovered known match wins; ambiguity blocks the
write. A successful request is consumed, so duplicate confirmation does not create another ID.
Names remain labels, so a confirmed different person may also be named Shrish. Neither a
`confirmed` field in LLM arguments nor a `CONFIRM_ENROLLMENT` LLM command can approve enrollment.

This is a backend contract, not a shipped wearer-facing confirmation flow. Automatic new-contact
creation is now off. An application adapter must obtain an explicit operator decision and pass the
current request ID; the glasses/dashboard confirmation UI is not included in this change.

## Run in VS Code

Open this fork, then use **Tasks: Run Task → Run identity situations and regression tests**.
The task runs the 16 scenarios, worker helpers (including 52 alias replays), database suite, and
fake-client API queue suite. Logs and a summary are saved in `.test-results/identity/`, ignored by
Git. The current result folder is recorded in `.test-results/identity/latest-run.txt`.

Equivalent Windows command from the fork root:

```powershell
.\.venv\Scripts\python.exe scripts/run_identity_checks.py
```

## Limits

These tests inject recognition results and registration intentions; SQLite is real but temporary.
They validate control flow, not face/voice accuracy, ASR, live timing, or UF intent interpretation.
Requiring confirmation prevents this automatic duplicate path; an incorrect human confirmation
or a false-positive biometric match can still be wrong. Existing duplicate records are untouched.
The same-recording reference and injected-intent limitations of the earlier recording tests remain.
