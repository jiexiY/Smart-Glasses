# Issue 8: identity registration investigation and tests

Current follow-up: [uncertain recognition tests](uncertain-identity-tests.md) removes automatic
enrollment after repeated unknown observations. The historical fix below is superseded on that point.

Upstream: https://github.com/UF-Smart-Glasses/Smart-Glasses/issues/8

Baseline: `455c04fb78089d33f70b02bb4b999ffa48eb7834`.
Work is isolated on the fork's `fix/identity-before-name` branch. No upstream changes or PR.

## Reproduced before changing the implementation

Added a regression to `tests/test_worker_helpers.py` with:

- an existing UUID named `jiexi yang`;
- a vision-cache result already recognizing that UUID;
- a registration intent naming the person `jess`.

Running the baseline handler created a second user named `jess` and failed:

```text
AssertionError: A recognized face must not create a second UUID
```

This proves the coordinator's missing-name branch can duplicate an identity even when supplied a
successful face-recognition result. It is an offline control-flow reproduction, not a live camera
or speech-recognition benchmark.

Follow-up: [`verified-name-learning.md`](verified-name-learning.md) documents alias learning on
`feat/verified-name-learning`, including changes to unknown-voice binding. The section below describes
the earlier registration fix.

## Change

- Resolve the target's face / self-introduction voice independently of the spoken name.
- Reuse the matched UUID and preserve its canonical name; no alias-table migration.
- Remove the fallback that selected the first same-name user without biometric agreement.
- Retry unresolved single-track requests automatically; never create on expiry.
- Require repeated valid unknown-face observations before creating a contact.
- Recheck the latest target, users and voices before the write.
- Keep an in-flight track-to-UUID mapping while workers update their embeddings.
- Continue using `REGISTER_FACE` / `REGISTER_VOICE` for worker-owned writes.

## Offline test setup

Use Python 3.11 in `.venv`. Only install the lightweight test dependencies:

```text
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
PYTHONPATH=. .venv/bin/python tests/test_worker_helpers.py
PYTHONPATH=. .venv/bin/python tests/test_database.py
PYTHONPATH=. .venv/bin/python tests/test_api_worker_queue.py
```

Windows PowerShell, after creating the Python 3.11 environment:

```powershell
$env:PYTHONPATH = '.'
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe tests/test_worker_helpers.py
.\.venv\Scripts\python.exe tests/test_database.py
.\.venv\Scripts\python.exe tests/test_api_worker_queue.py
```

All three suites passed on Windows with Python 3.11.15 on November 2, 2026. No OpenAI calls,
camera access or live identity database were used. The regression now passes. The SQLite
regression checks that renaming Shrish, then using the old name / surname / nickname, leaves
exactly one user with the original UUID and unchanged face, voice and chat records.

## Data already committed to this repository

| Files | Verified contents | Suitable use |
| --- | --- | --- |
| `api/simulator_resources/riley_self_intro_92.mp4` and `riley_self_intro.wav` | Readable 640x480 video, 193 frames at 9.2 fps (~20.98 s); 16 kHz mono 16-bit audio (~21.12 s) | Self-introduction replay |
| `api/simulator_resources/shrish_introduced_92.mp4` and `shrish_introduced.wav` | Readable 640x480 video, 201 frames at 9.2 fps (~21.85 s); 16 kHz mono 16-bit audio (~21.79 s) | Third-person introduction replay; currently selected by simulator config |
| `database/sample_data/voice_Matt1.npy`, `voice_Shaun1.npy`, `voice_Tiger1.npy`, `voice_Tiger2.npy` | Four finite float32 vectors, each shape `(192,)` | Existing-voice identity reuse regression; all four used by the offline test |
| `workers/audio_utils/solo_voices/` | Four committed WAV files: Matt, Shaun, TigerGroupChat, TigerTestOne | Source audio for voice-development work |

The real voice-vector test matches each vector to a stored copy. It checks UUID reuse, not
recognition accuracy across unseen recordings. Face regressions supply simulated recognition
results; there are no committed face-embedding reference fixtures in `database/sample_data`.
The video/audio pairs have slightly different durations. Do not assume frame-perfect alignment
or treat them as a labeled nickname/deduplication benchmark without preparing annotations.

## EasyCom branch

The repository's `easycom-hf-data` branch contains `scripts/fetch_easycom.py`:

https://github.com/UF-Smart-Glasses/Smart-Glasses/blob/easycom-hf-data/scripts/fetch_easycom.py

It downloads `sources/easycom/session_01` from the Hugging Face dataset
`Smart-Glasses/smart-glasses-eval`, copies one MP4, WAV and voice NPY into the simulator/sample
folders, and rewrites the simulator config. The script is not on upstream `main`.

An unauthenticated metadata request returned HTTP 401 on November 2, 2026. Its contents and
access requirements could not be verified; the response alone does not establish whether it
is private, gated or otherwise unavailable. No credentials were guessed and no data was downloaded.
The script writes its download cache to `~/sg-data`; inspect/adapt that path before running it.

Official EasyCom dataset and labels:
https://github.com/facebookresearch/EasyComDataset

EasyCom includes egocentric video, multichannel audio, transcriptions, target-of-speech labels,
and participant-linked face boxes. It is useful for later recognition/association evaluation,
but needs adaptation to this simulator's 16 kHz mono input and to the specific rename scenario.
The official suggested held-out test sessions are 10 and 11; session 1 is in its training split.

## Limits and next validation

- The full app and simulator were not run. `parakeet-mlx` depends on MLX; this Windows offline
  environment does not run that speech pipeline. No model-accuracy or end-to-end claim is made.
- Three frames / 0.5 seconds, a one-second tracking gap and a five-second retry window are
  initial controls. Measure them using labeled replay, including ASR/LLM delays, before deployment.
- Multiple possible target faces, unknown voice-only introductions, stale frames and expired
  requests can remain unresolved. The patch deliberately does not guess a person or create on timeout.
- Conflicting voice/face evidence stays unresolved until a later valid request or expiry; the
  pending request retains its original voice evidence and never switches to another face track.
- This is not a cross-process database uniqueness constraint on face embeddings. False-negative
  recognition, track switches, out-of-band database edits and queued worker failures still need
  end-to-end evaluation. Existing voice-association heuristics are outside this patch.
- Existing duplicate records are not merged or deleted. Original identity records are preserved.

For a hardware/model replay, capture the actual event stream, rename the enrolled person, then
repeat the same encounter with their full name, name parts and a nickname. Compare user count,
UUIDs and linked records before/after. A reliable known match must not create a new contact.
