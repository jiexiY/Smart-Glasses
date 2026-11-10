## Event Schema Documentation

The `Coordinator` processes an `event` dictionary received from the `results_queue`. Every event contains a `"type"` key that determines its structure.

### 1. Vision Result (`vision_result`)

Sent by the Vision worker to report detected faces and tracking data.

* **Source:** Vision Worker
* **Purpose:** Updates the temporal face cache and triggers identity registration

| Key | Type | Description |
| --- | --- | --- |
| `type` | `string` | Always `"vision_result"`. |
| `faces` | `list[dict]` | A list of detected face objects |
| `timestamp` | `float` | Timestamp when the face was processed |

**Face Object Structure:**

```python
{
    "track_id": int,          # uniqye tracking id for the face
    "bbox": (x1, y1, x2, y2), # bounding box for the face
    "name": string,           # identified name or default unknown name
    "score": float,          # confidence score with the associated name
    "emb": np.array | None    # embedding; only an np array when re-calced embedding
}

```

---

### 2. Speech/Audio Event (`speech`)

Sent by the Audio worker when human speech parsed

* **Source:** Audio/STT Worker
* **Purpose:** Provides text for intent parsing and potential identity verification via voice

| Key | Type | Description |
| --- | --- | --- |
| `type` | `string` | Always `"speech"`. |
| `text` | `string` | The transcribed text |
| `id` | `string` | The session ID (Shaun knows what this is) |
| `speech_start_time` | `float` | Time when sentence began |
| `final` | `bool` | pretty much always true; true if end of sentence |
| `embedding` | `np.array` | embedding for speaker voice |

---

### 3. VLM Result (`vlm_result`)

Sent when the Vision-Language Model completes a complex visual reasoning task

* **Source:** VLM Worker
* **Purpose:** Provides high-level semantic descriptions of the video feed.

| Key | Type | Description |
| --- | --- | --- |
| `type` | `string` | Always `"vlm_result"` |
| `request_id` | `int` | Maps the response back to a specific `Coordinator` request |
| `text` | `string` | The natural language description/answer from the VLM. |
| `timestamp` | `float` | Time the inference was completed |

---

### 4. Intent Event (`intent`)

Generated after the LLM parses a user's speech into a specific actionable command; individual 
`cmd` and `args` defined in `core.config_tools`

* **Source:** LLM/Logic Worker
* **Purpose:** Triggers specific state changes like registering a new user.

| Key | Type | Description |
| --- | --- | --- |
| `type` | `string` | Always `"intent"` |
| `cmd` | `string` | The command type (e.g., `REGISTER_FACE`, `CHAT`) |
| `args` | `dict` | Key-value pairs containing arguments for the command.|

---

### 5. Internal Commands (Outgoing)

The Coordinator sends these to the `commands_queue` to control sub-workers.

* **`GET_VIDEO_CONTEXT`**: Requests the VLM to analyze the current video buffer.
* **`REGISTER_FACE`**: Tells the Vision worker to bind a `track_id` to a specific `name` and save the embedding.



### 6. API Command Queue (`llm_command_queue`)

Commands sent to `APIWorker`:

- `PARSE_INTENT`
  - Payload: `{"cmd":"PARSE_INTENT","text":str,"timestamp":float,"voice_embedding":np.array|None}`
  - Result event: `intent`

- `ANALYZE_VIDEO_FRAMES`
  - Payload: `{"cmd":"ANALYZE_VIDEO_FRAMES","frames":list[bytes],"prompt":str,"request_id":int}`
  - Result event: `vlm_result`

- `ANALYZE_MEMORY`
  - Payload: `{"cmd":"ANALYZE_MEMORY","conversation_history":str,"known_facts":str,"subject":str}`
  - Result event: `memory_result`

### Name-learning evidence

`PARSE_INTENT` commands and returned intents carry `name_source`: original utterance `event_id`, `text`, `timestamp`, `voice_user_id`, `voice_score`, `voice_margin`, and `face_user_ids`. The coordinator computes this snapshot from stored profiles when speech arrives. The API helper copies it from the command, not the LLM response. Voice scores are similarities, not calibrated probabilities. Missing/stale source metadata cannot teach an alias.

The snapshot also includes `face_count` (all faces, including unknowns), `face_timestamp` (the selected
coordinator-cache timestamp, or null), `face_score` (finite score for a sole face, or null), and
`face_user_id` (sole known face above the matching threshold, or null). A snapshot without these
verified face fields cannot authorize a new alias. Cache time is not a device capture timestamp;
this change does not add active-speaker detection or independent face-model observations.

Speech `id` is a full utterance UUID and `time_start` is its start timestamp. Known-person name learning uses these to deduplicate events. See `docs/verified-name-learning.md`.

### 7. API Error

When there is an error with any api call

| Key | Type | Description |
| --- | --- | --- |
| `type` | `string` | Always `"api_error"` |
| `error` | `string` | The error in string format |
| `timestamp` | `time` | Time stamp when error happened.|

### 8. Local enrollment confirmation (`confirm_enrollment`)

This event belongs to an explicit operator confirmation adapter, not the LLM tool schema.
The pending request is logged by the coordinator and bound to its original name, face track and
five-second lifetime. Repeated unknown observations never count as confirmation.

```python
{"type": "confirm_enrollment", "request_id": "pending-request-uuid", "confirmed": True}
```

Only the exact live request can be confirmed. A fresh sole target, stable observations, and a
new identity check are required. Known matches reuse their UUID; conflicts, expiry, stale frames,
or changed targets do not enroll. Duplicate confirmations are harmless. The LLM cannot trigger
this by emitting `CONFIRM_ENROLLMENT` or adding `confirmed` to `REGISTER_IDENTITY` arguments.
There is no wearer-facing confirmation UI in this patch; automatic new-contact creation is off.
