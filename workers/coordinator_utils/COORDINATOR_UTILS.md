# coordinator_utils

Event handling and voice/face association for the coordinator. `Coordinator.run` only pulls events and calls `handle_event(state)`.

## handlers.py

Speech, intent, and identity registration. `CoordinatorState` holds the db (`IdentityDirectory`), vision and audio caches, conversation history, pending voice registration, and the llm, vision, and audio command queues.

`vision_result` caches faces. `speech` binds the speaker, then enqueues `PARSE_INTENT`. `intent` runs `REGISTER_IDENTITY`, `SPEAK`, or `VISION_CONTEXT`. `vlm_result` and `api_error` log.

### Speech

An unknown voice (`config.DEFAULT_ID` plus an embedding) is cached. A pending registration inside the window (`max_delay`) binds that user and enqueues `REGISTER_VOICE`. Otherwise visual association may bind a known face and register the voice. Known users get chat history saved. Every utterance is appended and `PARSE_INTENT` is enqueued with recent history and the names currently in view.

### Identity

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

An unknown track must have at least three valid observations spanning 0.5 seconds, with no gap over
one second, before enrollment. A final check rereads users/voices and checks the latest frame. The
latest frame must still contain the original sole target. These are conservative initial controls,
not calibrated recognition-accuracy guarantees. Recognition still relies on the vision worker's
face result; the coordinator does not implement a second face model.

`registered_tracks` prevents another UUID while the owning worker is processing `REGISTER_FACE`.
It follows continuously observed tracks and expires after a one-second tracking gap. New enrollment
captures the selected embedding instead of selecting an unrelated unknown face later. The existing
unknown-voice association and post-introduction voice capture remain separate behavior.

### identity.py

Pure identity resolution, embedding validation, and observation-window checks. Runtime state types
are `PendingIdentity` and `RegisteredTrack`; `CoordinatorState` owns their dictionaries. Thresholds
come from the existing audio and vision config. No schema migration or stored alias records.

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
