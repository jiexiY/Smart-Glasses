import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from core import config
from workers.coordinator_utils.association import (
    PendingVoiceRegistration,
    trim_before,
)
from workers.coordinator_utils.identity import (
    MAX_FRAME_AGE,
    RETRY_WINDOW,
    PendingIdentity,
    RegisteredTrack,
    resolve_identity,
    stable_unknown_track,
    valid_embedding,
)
from workers.coordinator_utils.name_learning import capture_name_source, learn_name


class QueueWriter(Protocol):
    def put_nowait(self, obj: Any) -> None: ...


class IdentityDirectory(Protocol):
    def get_user_name(self, user_id: str) -> str: ...
    def save_chat_history(self, user_id: str, transcript: str) -> None: ...
    def get_user_ids_by_name(self, name: str) -> list[str]: ...
    def get_voice_embeddings_by_uid(self, user_id: str) -> list[np.ndarray]: ...
    def get_all_users(self) -> dict[str, str]: ...
    def get_all_voices(self) -> dict[str, list[np.ndarray]]: ...
    def create_user(self, user_id: str, name: str) -> None: ...
    def record_name_observation(self, **observation: Any) -> dict[str, Any]: ...


@dataclass
class CoordinatorState:
    db: IdentityDirectory
    llm_commands: QueueWriter
    vision_commands: QueueWriter
    audio_commands: QueueWriter
    vision_cache: deque[tuple[float, list[dict[str, Any]]]]
    audio_cache: deque[tuple[float, np.ndarray]]
    conversation_history: deque[str]
    pending_voice_registration: PendingVoiceRegistration | None
    cache_duration: float
    audio_cache_duration: float
    max_delay: float
    frame_area: float
    frame_center_x: float
    frame_center_y: float
    max_frame_distance: float
    request_number: int
    pending_identities: dict[int, PendingIdentity] = field(default_factory=dict)
    registered_tracks: dict[int, RegisteredTrack] = field(default_factory=dict)
    name_learning_mode: str = field(default_factory=lambda: config.NAME_LEARNING_MODE)
    name_decisions: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=100))


def handle_event(event: dict[str, Any], state: CoordinatorState) -> None:
    handlers: dict[str, Callable[[dict[str, Any], CoordinatorState], None]] = {
        "vision_result": handle_vision_result,
        "speech": handle_speech,
        "vlm_result": handle_vlm_result,
        "intent": handle_intent,
        "api_error": handle_api_error,
    }
    handler = handlers.get(event.get("type", "unknown"), handle_unknown_event)
    handler(event, state)


def handle_vision_result(event: dict[str, Any], state: CoordinatorState) -> None:
    # keep a short history of who was in frame
    now = time.time()
    state.vision_cache.append((now, event.get("faces", [])))
    trim_before(state.vision_cache, now, state.cache_duration)
    faces = event.get("faces", [])
    active_tracks = {face.get("track_id") for face in faces}
    for track_id, registration in list(state.registered_tracks.items()):
        if now - registration.last_seen > MAX_FRAME_AGE:
            del state.registered_tracks[track_id]
        elif track_id in active_tracks:
            registration.last_seen = now
    retry_pending_identities(state, now, faces)


def handle_speech(event: dict[str, Any], state: CoordinatorState) -> None:
    user_id = event.get("user_id", config.DEFAULT_ID)
    text = event.get("text", "")
    timestamp = event.get("time_start", time.time())
    voice_embedding = event.get("embedding")
    # snapshot before any heuristic voice binding or asynchronous LLM work
    name_source = capture_name_source(
        event, state.db.get_all_users(), state.db.get_all_voices(), state.vision_cache
    )

    # being visible or speaking next does not prove which person owns this voice
    if user_id == config.DEFAULT_ID and voice_embedding is not None:
        remember_unknown_voice(state, timestamp, voice_embedding)

    # save the line and ask the llm what they just meant
    speaker_name = display_name(user_id, state.db.get_user_name)
    record_transcript(state, user_id, speaker_name, text)
    state.llm_commands.put_nowait(
        {
            "cmd": "PARSE_INTENT",
            "text": intent_prompt(state, speaker_name),
            "timestamp": timestamp,
            "voice_embedding": voice_embedding,
            "name_source": name_source,
        }
    )
    print(f"[Coordinator] {speaker_name}: {event['text']}")


def handle_vlm_result(event: dict[str, Any], state: CoordinatorState) -> None:
    print(f"[Coordinator] Received VLM output: {event['text']}")


def handle_intent(event: dict[str, Any], state: CoordinatorState) -> None:
    command = event.get("cmd", "CHAT")
    args = event.get("args", {})
    if not isinstance(args, dict):
        return
    timestamp = event.get("timestamp", time.time())
    voice_embedding = event.get("voice_embedding")

    # llm told us to add a person
    if command == "REGISTER_IDENTITY":
        source = event.get("name_source")
        if isinstance(source, dict):
            decision = learn_name(state.db, args, source, time.time(), state.name_learning_mode)
            state.name_decisions.append(decision)
            print(f"[Coordinator] Name learning: {decision}")
            # learning an alias must not also teach a face or voice from the same guess
            if decision.get("user_id") is not None or decision.get("status") == "ignored":
                return
        register_identity(state, args, timestamp, voice_embedding)
    elif command == "SPEAK":
        message = args.get("message", "")
        if message:
            print(f"[Coordinator]: [STEVE]: {message}")
    # ask vision to describe what's in frame
    elif command == "VISION_CONTEXT":
        state.vision_commands.put_nowait(
            {
                "cmd": "GET_VIDEO_CONTEXT",
                "prompt": args.get("prompt", "Summarize the video in a sentence"),
                "request_id": state.request_number,
            }
        )


def handle_api_error(event: dict[str, Any], state: CoordinatorState) -> None:
    print(f"Error: {event.get('error')}\nTime: {event.get('timestamp')}")


def handle_unknown_event(event: dict[str, Any], state: CoordinatorState) -> None:
    print("\n[Coordinator] got other event")


def display_name(user_id: str, name_for: Callable[[str], str]) -> str:
    if user_id == config.DEFAULT_ID:
        return config.DEFAULT_NAME
    return name_for(user_id)


def remember_unknown_voice(
    state: CoordinatorState, timestamp: float, voice_embedding: np.ndarray
) -> None:
    now = time.time()
    state.audio_cache.append((timestamp, voice_embedding))
    trim_before(state.audio_cache, now, state.audio_cache_duration)


def record_transcript(
    state: CoordinatorState, user_id: str, speaker_name: str, text: str
) -> None:
    transcript = f"{speaker_name}: {text}"
    if text and user_id != config.DEFAULT_ID:
        state.db.save_chat_history(user_id, transcript)
    state.conversation_history.append(transcript)


def intent_prompt(state: CoordinatorState, speaker_name: str) -> str:
    active_names = names_in_view(state)
    vision_context = (
        f"Known people in view: {list(active_names)}\n" if active_names else ""
    )
    history_text = "\n".join(state.conversation_history)
    return (
        f"{vision_context}"
        f"Transcript:\n{history_text}\n\n"
        f"Task: Parse the intent for the latest message from {speaker_name}."
    )


def names_in_view(state: CoordinatorState) -> set[str]:
    if not state.vision_cache:
        return set()
    _, faces = state.vision_cache[-1]
    return {
        state.db.get_user_name(uid)
        for face in faces
        if (uid := face.get("user_id", config.DEFAULT_ID)) != config.DEFAULT_ID
    }


def register_identity(
    state: CoordinatorState,
    args: dict[str, Any],
    timestamp: float,
    voice_embedding: np.ndarray | None,
) -> None:
    raw_name = args.get("name", "")
    if not isinstance(raw_name, str):
        return
    name = raw_name.strip()
    if not name or name.casefold() == config.DEFAULT_NAME.casefold():
        return
    pending = PendingIdentity(dict(args), timestamp, voice_embedding, -1, timestamp)
    attempt_identity_registration(state, pending, timestamp)


def attempt_identity_registration(
    state: CoordinatorState, pending: PendingIdentity, timestamp: float
) -> None:
    name = pending.args["name"].strip()
    is_self_intro = pending.args.get("is_self_introduction", False)
    users = state.db.get_all_users()
    voices = state.db.get_all_voices()
    resolution = resolve_identity(
        state.vision_cache, timestamp, users, voices,
        pending.voice_embedding, is_self_intro, state.registered_tracks,
    )
    face = resolution.face
    track_id = face.get("track_id") if face else None
    if pending.track_id != -1 and track_id != pending.track_id:
        return
    if resolution.status == "known" and resolution.user_id is not None:
        reuse_identity(state, pending, resolution.user_id, face, users, voices, timestamp)
        return

    if not isinstance(track_id, int):
        return
    previous = state.pending_identities.get(track_id)
    if previous is not None and previous is not pending:
        if 0 <= timestamp - previous.timestamp <= RETRY_WINDOW:
            attempt_identity_registration(state, previous, timestamp)
            return
        del state.pending_identities[track_id]
    pending.track_id = track_id
    if state.vision_cache:
        latest_time, latest_faces = state.vision_cache[-1]
        if any(item.get("track_id") == track_id for item in latest_faces):
            pending.last_seen = latest_time
    state.pending_identities.setdefault(track_id, pending)
    if resolution.status != "unknown":
        return
    # an existing name without biometric agreement stays unresolved
    if state.db.get_user_ids_by_name(name):
        return
    if not stable_unknown_track(state.vision_cache, track_id, timestamp):
        return
    # resolve again against current records before the first write
    latest_time, latest_faces = state.vision_cache[-1]
    if (len(latest_faces) != 1 or latest_faces[0].get("track_id") != track_id
            or not 0 <= latest_time - pending.timestamp <= RETRY_WINDOW):
        return
    latest_users = state.db.get_all_users()
    latest_voices = state.db.get_all_voices()
    latest = resolve_identity(
        state.vision_cache, latest_time, latest_users, latest_voices,
        pending.voice_embedding, is_self_intro, state.registered_tracks,
    )
    if latest.status == "known" and latest.user_id is not None:
        reuse_identity(state, pending, latest.user_id, latest.face, latest_users, latest_voices, latest_time)
        return
    if latest.status != "unknown" or name in latest_users.values():
        return
    if not stable_unknown_track(state.vision_cache, track_id, latest_time):
        return
    create_identity(
        state, name, pending.args.get("speaker_name", config.DEFAULT_NAME),
        is_self_intro, pending.timestamp, pending.voice_embedding, latest.face, latest_time,
    )
    state.pending_identities.pop(track_id, None)


def reuse_identity(
    state: CoordinatorState,
    pending: PendingIdentity,
    user_id: str,
    face: dict[str, Any] | None,
    users: dict[str, str],
    voices: dict[str, list[np.ndarray]],
    timestamp: float,
) -> None:
    track_id = face.get("track_id") if face else None
    if isinstance(track_id, int):
        state.pending_identities.pop(track_id, None)
        if (face.get("user_id", config.DEFAULT_ID) == config.DEFAULT_ID
                and valid_embedding(face.get("emb"))
                and track_id not in state.registered_tracks):
            state.registered_tracks[track_id] = RegisteredTrack(user_id, pending.timestamp, timestamp)
            state.vision_commands.put_nowait({
                "cmd": "REGISTER_FACE", "track_id": track_id,
                "user_id": user_id, "emb": face["emb"],
            })
        if (pending.args.get("is_self_introduction", False)
                and valid_embedding(pending.voice_embedding) and not voices.get(user_id)):
            enqueue_register_voice(state.audio_commands, user_id, pending.voice_embedding)
    print(f"[Coordinator] Reused identity: {users[user_id]} ({user_id}); heard '{pending.args['name']}'")


def retry_pending_identities(
    state: CoordinatorState, timestamp: float, faces: list[dict[str, Any]]
) -> None:
    for track_id, pending in list(state.pending_identities.items()):
        if (not 0 <= timestamp - pending.timestamp <= RETRY_WINDOW
                or timestamp - pending.last_seen > MAX_FRAME_AGE):
            del state.pending_identities[track_id]
            continue
        if any(face.get("track_id") == track_id for face in faces):
            pending.last_seen = timestamp
        if len(faces) == 1 and faces[0].get("track_id") == track_id:
            attempt_identity_registration(state, pending, timestamp)


def create_identity(
    state: CoordinatorState,
    name: str,
    speaker_name: str,
    is_self_intro: bool,
    timestamp: float,
    voice_embedding: np.ndarray | None,
    face: dict[str, Any],
    observed_at: float,
) -> None:
    user_id = str(uuid.uuid4())
    state.db.create_user(user_id, name)
    print(f"[Coordinator] Created new identity: {name} ({user_id})")
    state.registered_tracks[face["track_id"]] = RegisteredTrack(user_id, timestamp, observed_at)
    state.vision_commands.put_nowait({
        "cmd": "REGISTER_FACE", "track_id": face["track_id"],
        "user_id": user_id, "emb": face["emb"],
    })
    # they introduced themselves, so keep the voice that just spoke
    if (
        speaker_name == config.DEFAULT_NAME
        and is_self_intro
        and valid_embedding(voice_embedding)
    ):
        enqueue_register_voice(state.audio_commands, user_id, voice_embedding)


def enqueue_register_voice(queue: QueueWriter, user_id: str, embedding: np.ndarray) -> None:
    queue.put_nowait(
        {"cmd": "REGISTER_VOICE", "user_id": user_id, "embedding": embedding}
    )
