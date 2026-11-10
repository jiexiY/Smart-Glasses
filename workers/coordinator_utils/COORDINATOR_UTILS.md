# coordinator_utils

Event handling and voice/face association for the coordinator. `Coordinator.run` only pulls events and calls `handle_event(state)`.

## handlers.py

Speech, intent, and identity registration. `CoordinatorState` holds the db (`IdentityDirectory`), vision and audio caches, conversation history, pending voice registration, and the llm, vision, and audio command queues.

`vision_result` caches faces. `speech` keeps the audio worker's speaker result, then enqueues `PARSE_INTENT`. `intent` runs `REGISTER_IDENTITY`, `SPEAK`, or `VISION_CONTEXT`. `vlm_result` and `api_error` log.

### Speech

Unknown voices are cached but are no longer assigned to a visible person or the next speaker after a greeting. Known users get chat history saved. Each utterance snapshots original text, ID, timestamp, voice candidates and known faces before the asynchronous intent call. `name_source` travels through that call independently of LLM output.

### Identity

Known-person intents carrying `name_source` first enter `name_learning.py`, described below.
The remaining registration flow still handles unknown identities and legacy direct intents.

`REGISTER_IDENTITY` resolves biometrics before names. A single recent recognized face, or a unique
self-introduction voice match across all stored voices, reuses the existing UUID without renaming it.
First names, surnames, old names and arbitrary nicknames cannot create another UUID for that match.
The wearer's greeting voice is not used to identify the person being greeted.

Reliable voice recognition can attach the one unambiguous unknown face to the existing UUID. A
recognized face can add a self-introduction voice when that UUID has no stored voice yet. Both use
the existing worker command queues, so database and worker memory are updated by the owner.

`pending_identities` holds unresolved single-track requests, separate from `pending_voice_registration`.
Each subsequent vision event retries only the original track. Conflicting matches, multiple possible
voices, stale frames, invalid embeddings and unverified existing names never select a guessed UUID.
Requests expire after five seconds from the original utterance timestamp, or when their track has
not been observed for more than one second. Expiry does not create a user. No target track means no
automatic face enrollment; a new introduction or usable self-introduction voice is needed.

An unknown track never enrolls automatically, even after repeated observations. The separate local
`confirm_enrollment` event must contain `confirmed: true` and the pending request's `request_id`.
It is not an LLM intent/tool or a name argument. Confirmation is one-use and bound to the original
name, track and five-second lifetime. The latest frame must be fresh and contain only that track,
with three valid observations spanning 0.5 seconds and no gap over one second. A final identity
check reuses a recovered existing person or rejects conflicting evidence before any new write.
An explicitly confirmed new person can share an existing display name. The coordinator supports
this confirmation contract; a wearer-facing confirmation UI is not implemented here.
These are conservative controls, not calibrated recognition-accuracy guarantees. Recognition
still relies on the vision worker's face result; the coordinator has no second face model.

`registered_tracks` prevents another UUID while the owning worker is processing `REGISTER_FACE`.
It follows continuously observed tracks and expires after a one-second tracking gap. New enrollment
captures the selected embedding instead of selecting an unrelated unknown face later. Unknown-voice association based only on visibility or greeting timing has been removed from the speech path.

### identity.py

Pure identity resolution, embedding validation, and observation-window checks. Runtime state types
are `PendingIdentity` and `RegisteredTrack`; `CoordinatorState` owns their dictionaries. Thresholds
come from the existing audio and vision config.

### name_learning.py

Separate alias policy (`self-name-v2`): a whole explicit self-naming utterance, unique stored voice match, and exactly one recent face whose finite score exceeds the existing face threshold and whose UUID agrees with the voice. Unknown bystanders count toward the face count. Missing, weak, stale, multiple, or conflicting faces leave a known speaker's candidate pending. Two distinct eligible utterances from the same policy can confirm an alias in `auto` mode; default `shadow` records evidence without exposing aliases in lookup. Known-person name intents do not also enroll new biometric samples. `name_decisions` holds the latest 100 diagnostic decisions. Database methods store all candidate evidence. See `docs/verified-name-learning.md` for thresholds and limits, and `docs/name-learning-test-first.md` for failing-then-passing regressions.

## association.py

Timing and geometry helpers. No queues or db.

- `pending_registration_hit`: voice time is in `[0, max_delay)` after the pending registration.
- `closest_frame`: vision frame nearest the timestamp.
- `face_matches`: that uid is in the frame.
- `voice_matches`: self-intro embedding vs the mean stored voice, cosine similarity above 0.30.
- `associate_voice`: within 1s, a known face speaking in at least 30% of frames, or the only known person in at least 80%.
- `resolve_unknown_face`: unknown face with the highest attention score, `0.3 * area_scaled - 0.7 * distance_scaled`.
- `resolve_unknown_voice`: first cached unknown voice in `[0, max_delay]` after the target time.
- `trim_before`: drop cache entries older than the duration.

The visibility-only and next-speaker association helpers are retained for legacy helper tests,
but no longer authorize speech-handler voice enrollment. `pending_voice_registration` remains
in the state shape for compatibility and is no longer populated or consumed by this handler.
