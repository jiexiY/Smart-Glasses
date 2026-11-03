# Verified name learning: first implementation

Branch: `feat/verified-name-learning`, based on `fix/identity-before-name`.
This is an evaluation implementation, not a production accuracy claim.

## What runs

1. On speech arrival, `capture_name_source` snapshots the original transcript, full utterance
   UUID, utterance timestamp, visible known identities, and a voice match against stored profiles.
   It does not trust the LLM to identify the speaker. This snapshot travels through `PARSE_INTENT`
   and returns unchanged in `name_source`; LLM output cannot replace it.
2. The LLM proposes a registration/name intent. The coordinator independently requires a whole,
   explicit self-naming utterance, such as `My name is Shri`, `Call me Shri`, or `I go by Shri`.
   Negations, quotations, ordinary mentions, and introductions by another person do not qualify.
   Names are preserved as transcribed rather than silently replaced with a saved canonical name.
3. Automatic alias eligibility requires a unique saved-voice match **and exactly one recent face**
   with the same UUID and a finite score above the existing face threshold (currently 0.5).
   The selected frame must be within one second of the utterance start. Unknown bystanders also
   count as faces. Voice-only, face-only, weak/stale face results, multiple faces, and disagreements
   stay pending when there is a candidate identity. This conservative rule reduces automatic
   coverage; it does not establish that the visible person is speaking.
4. A transaction records the observation and keeps the alias pending until two eligible utterances
   with distinct IDs and timestamps support the same person/name within five minutes, at least
   one second apart. Replayed events cannot vote twice, even across process restarts or concurrent
   calls. One utterance cannot vote for two names or people. Both votes must use the same policy
   version (`self-name-v2`); old voice-only evidence cannot count toward this stricter rule.
5. In `auto` mode, an eligible candidate becomes a confirmed alias. The canonical name, UUID,
   existing biometric records and conversation history stay attached to the same person.

## Default and configuration

`NAME_LEARNING_MODE=shadow` is the default. It records pending candidates and evidence, and reports
`would_confirm` when the policy would accept an alias. It does not put those aliases into name lookup.

`NAME_LEARNING_MODE=auto` permits confirmation. Set the environment variable before starting a
supported app environment. This mode is exercised by offline tests here, not enabled on a live app.
Switching modes does not retroactively accept shadow observations without another eligible event.

Shadow mode applies **only to alias learning**, not the pre-existing new-person enrollment code.
Both modes write pending/audit rows. Neither changes the existing canonical name.

Initial evaluation controls in `name_learning.py`: voice similarity at least 0.70, runner-up gap
at least 0.15, and intent no older than 60 seconds. These are provisional, uncalibrated thresholds,
not probabilities. The two-utterance/five-minute rule is also an uncalibrated operating policy.

## Database and corrections

The additive migration creates `identity_names` and `name_observations` without replacing users.
Aliases are unique per `(user_id, normalized_name)`, not globally: two people can both be called Sam.
Evidence includes the source snapshot and policy version. Raw embeddings are not copied into the
name audit. Transactions serialize confirmation, and event IDs deduplicate observations.
Already confirmed aliases are preserved on upgrade; review previously confirmed names separately
if the older automatic mode was used. This change does not silently revoke them.

- `get_user_ids_by_name`: canonical names and confirmed aliases, normalized for case/spacing.
- `find_user_candidates`: full names or individual space-separated name parts. Results are only
  candidates; no identity is selected or merged based on these results.
- `get_identity_names`: inspect pending, confirmed and rejected names for a person.
- `reject_identity_name`: remove an alias from lookup while retaining its evidence. Later automatic
  observations cannot reactivate a rejected alias. There is no correction UI in this increment.

Deletion of a user cascades to its aliases and observations. The repository's existing
`START_WITH_SAMPLE_DATA=True` behavior clears **all** identities on app shutdown, including aliases.
Persistent evaluation requires disabling that existing development flag and using a separate test
database. This work never opened or modified the live `database/identities.db`.

## Preventing unverified voice updates

The speech handler no longer saves an unknown voice merely because one known face is visible or
because that voice spoke after a greeting. Those heuristics could label a listener with someone
else's voice. Unknown speech stays unknown unless the existing audio recognizer identifies it.
Known-person name-learning intents do not also enqueue face/voice enrollment from the same guess.

## Replay and tests

From the clone in PowerShell:

```powershell
$env:PYTHONPATH = '.'
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe tests/test_worker_helpers.py
.\.venv\Scripts\python.exe tests/test_database.py
.\.venv\Scripts\python.exe tests/test_api_worker_queue.py
.\.venv\Scripts\python.exe -m scripts.replay_name_learning --mode shadow
.\.venv\Scripts\python.exe -m scripts.replay_name_learning --mode auto
```

`tests/fixtures/name_learning.json` contains 16 hand-labeled synthetic scenarios, and
`tests/fixtures/name_learning_agreement.json` adds 10 agreement regressions. Replay runs the
speech handler, source snapshot, fake intent output, coordinator and actual temporary SQLite DB.
It covers nicknames, shared names, duplicates, overlapping timestamps, stale results, expired
evidence, off-camera/unknown speech, conflicting faces, greetings, mentions, negations and quotes.
Reports count correct/wrong/missed aliases, added user records, and saved aliases. Each scenario
also asserts its expected pending/confirmed behavior. Both modes run in the approved helper suite.
Database tests also cover migration, persistence, concurrent events, correction and cascade deletion.

These results validate control flow with fixed fake model outputs. They do not measure face,
voice, speech transcription, LLM intent accuracy, automatic coverage in real conversations, latency,
or battery use. No model, camera, OpenAI call or real identity database is used in these runs.

Validation on November 3, 2026: all three offline suites passed. All 26 replay scenarios passed
in each mode (52 scenario runs); expected alias decisions matched, no unexpected user records
were created, and existing names and biometric profiles were unchanged in these synthetic cases.
The new checks were run and failed before changing the implementation; see
[`name-learning-test-first.md`](name-learning-test-first.md) for before/after evidence.

## Required before a production rollout

- Gather separately recorded, labeled enrollment and evaluation encounters with the actual glasses.
  Include unknown people, overlapping speech, quiet speakers, low light, similar voices, different
  names, accents, device motion and changes in the number of saved people.
- Replay recorded model events using the JSON observation contract, preserving source timestamps
  and utterance IDs. The current runner supplies the intent outputs; a full model harness must also
  capture and evaluate the ASR/LLM outputs rather than assuming they are correct.
- Calibrate voice thresholds and confirmation policy on development data, then evaluate on held-out
  sessions. Report false alias assignments and missed aliases alongside automatic coverage and delay.
- Implement and validate active-speaker/target-of-speech association before learning from greetings
  or third-party introductions. Existing face/voice scores alone do not prove name ownership.
- Improve new-person enrollment separately. The existing stable-unknown-face rule is still a
  heuristic; an unknown face result does not prove a previously unseen person. Legacy direct intent
  callers without `name_source` still use the earlier registration flow, without alias learning.
- Validate the trustworthiness of enrolled voice profiles; two repeated false voice matches are
  still wrong. Temporal repetition is not a mathematical guarantee of independent evidence.
  The vision worker also caches recognized identities; two utterances can reuse one model result.
  Frame freshness here uses coordinator receipt time, not the device capture clock. Calibrating
  thresholds, independent observations and actual speaker-to-face attribution remain separate work.
- Add retention controls, a correction UI, model/profile versioning, and staged live evaluation.
  Pending observations are retained for inspection; only the last five minutes qualify for a new
  confirmation. No automatic cleanup or monitoring service is installed in this increment.

The existing simulator clips and optional EasyCom source are catalogued in
[`identity-registration-testing.md`](identity-registration-testing.md). They need additional labels
and held-out recordings before they can substantiate production accuracy.
