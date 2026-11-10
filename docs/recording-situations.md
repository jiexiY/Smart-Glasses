# Recording-based identity situations

`scripts/replay_recording_situations.py` reruns the actual InspireFace, RedimNet and Windows ONNX
Parakeet models on the repository's Riley and Shrish recordings. It creates blurred, pixelated,
recovering, and noisy copies locally. Model outputs are not overwritten with pretend matches.
The resulting recognition events and transcribed name candidates are replayed into isolated SQLite
databases using both the pre-fix handler (`2e44f9f`) and the current handler.

## Cases

| Case | Controlled change | Expected behavior |
| --- | --- | --- |
| Saved face, different spoken label | Saved label Shrish; actual recorded ASR output “Hey Shri.” | Keep existing UUID |
| Blurred face, known-voice control | Gaussian blur, 31-pixel kernel; enrollment reuses the intro audio | Keep existing UUID |
| Blurred face, weak voice reference | Same blur; a separate recorded utterance is the fixture's voice reference | Wait or recover; never auto-create |
| Pixelated face, weak voice reference | Reduce to 80×60, then resize to 640×480 | Same safeguard |
| Blur clears at six seconds | Return to clean video without changing any recognition outputs | Recover the saved UUID within the pending window |
| Different face, same label | Use the other recording's face reference but give it display name Riley | Create a second UUID only after explicit confirmation |
| Moderate audio noise | Seeded white noise at +10 dB SNR, with blurred video | Preserve saved identity despite transcription differences |
| Severe audio noise | Seeded white noise at −10 dB SNR, with blurred video | No automatic duplicate; report missing speech separately |

Face references: Riley at/after 5 seconds; Shrish at/after 0.109 seconds. Speech windows: Riley
3.36–4.96 seconds; Shrish 3.2–4.32 seconds. The weak-voice reference is Riley-clip audio at
12–15.84 seconds. These choices are fixed and reported. The initial exploratory probes tried
several image degradations and voice references; these are development cases, not a held-out
benchmark. Face and voice thresholds remain at the repo's configured values.

The known-voice control deliberately reuses the enrollment utterance, so it cannot establish
general speaker-recognition accuracy. The weak-voice sample belongs to an unverified speaker
turn assigned by the fixture; a low score is not proof of a speaker-recognition error.
The same-name label is constructed in the test database, not an assertion about either person's name.

## Run

In VS Code: **Tasks: Run Task → Run recording situations (real models)**.

The current Windows setup already has the models cached under the enclosing task's `work/`
directory. Other checkouts can supply paths explicitly:

```powershell
.\.venv\Scripts\python.exe scripts/replay_recording_situations.py --face-model C:\models\Megatron --asr-model C:\models\parakeet-tdt-v3-int8
```

Additional installed packages are `inspireface==1.2.4.post3`, `onnxruntime==1.31.1`, and
`onnx-asr==0.12.0`, alongside the existing test dependencies. The ASR directory contains the
INT8 encoder/decoder, config, vocabulary and preprocessor from
`istupakov/parakeet-tdt-0.6b-v3-onnx`, revision `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`.
This script uses local files; it does not upload recordings or call a remote AI service.

Each run writes to ignored `.test-results/recording-situations/<UTC-time>/`:

- `results.json`: scores, exact transcripts, per-case coverage/outcome, model/input hashes and versions.
- `summary.md`: the before/after table.
- `*-faces.json`: actual timestamped recognition outputs without raw embedding vectors.
- `media/`: transformed video, tested utterance WAVs, references and the weak-voice sample.

MP4 files have video only; each WAV contains just the tested utterance, whose original timing is
in the JSON. `latest-run.txt` points to the latest completed run. A passing safety check does not
mean speech was transcribed correctly: a case with no extracted name skips registration and is
marked as not exercising that branch. Each of the four core situations separately requires its
recognition condition to have actually occurred before it can pass.

## Limits

The registration instruction is supplied by a small test adapter after transcription; it is not
a live UF AI intent result. New-person confirmation is injected by the harness. Vision and audio
timestamps use file-zero origins, and intent arrival is set to utterance end plus 0.2 seconds.
Original VAD segment boundaries are reused; noisy audio is not passed through VAD again.
No live worker queues, model feedback after enrollment, original Mac MLX backend, real glasses,
or independently captured enrollment encounters are tested. References are from the same clips.
The temporary databases are separate from the live identity database.

## Verified run

Ran through the VS Code task on November 9, 2026 (Eastern), with output folder
`2026-11-10T03-04-19-100365Z` (UTC). The tested implementation was `81132bf`, compared with
the earlier fork implementation `2e44f9f`. All four core conditions occurred and passed.

| Situation | Earlier fork: people stored | Fixed fork: people stored | Observed result |
| --- | --- | --- | --- |
| Saved face, different name | 1 | 1 | Kept the saved person |
| Blurred face, known-voice control | 1 | 1 | Kept the saved person |
| Blurred face, weak voice reference | 2 | 1 | Waited, then expired without a duplicate |
| Pixelated face, weak voice reference | 2 | 1 | Same safeguard held |
| Blur clears | 1 | 1 | Recovered the saved person before expiry |
| Different face, same name | 1 | 2 | Created a separate person after injected confirmation |
| Moderate noise | 2 | 1 | No duplicate; ASR changed the intro to “Hey, I'm right there.” |
| Severe noise | 1 | 1 | Empty transcript; registration was not exercised |

The weak-voice similarity was about 0.296, below the unchanged 0.30 threshold. The known-voice
control scored 1.0 because it reused the same utterance. These numbers explain the test paths;
they are not recognition accuracy rates. The existing person's saved name and history remained
unchanged in every case. Raw run evidence remains local under the ignored results directory.
