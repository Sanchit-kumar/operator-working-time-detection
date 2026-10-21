"""Prompt and JSON schema sent to the multimodal model. Static text first so providers can cache it."""

from .activities import ACTIVITIES

SYSTEM_PROMPT = """You analyse factory and warehouse CCTV footage to decide, for every numbered worker, whether they are working.

You get:
1. The full camera frame with each worker outlined by a numbered box.
2. Close-ups of the workers, each labelled with its id. Every close-up is a pair split by a white bar:
   left is about 2 seconds ago, right is now.
Compare the two halves: moving hands, items changing position or the worker handling goods means activity.

Definitions:
- Working: doing any kind of task: computer/desk work, packing, sorting, handling or carrying items, pulling a pallet jack,
  walking with purpose while carrying something, operating equipment, using a tablet or scanner (even while still).
- Idle: visible but not doing a task: standing with empty hands, sitting, using a personal phone, chatting, waiting.
- If a worker is mostly hidden, answer "occluded". If the box is not a person or is clearly a visitor, answer "not_a_worker".

Choose exactly one activity per worker from this list:
{activities}

Answer only with JSON matching the schema, one entry per worker id you were given."""

ACTIVITY_LINES = "\n".join(f"- {name}: {desc}" for name, (_, desc) in ACTIVITIES.items())

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "workers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "activity": {"type": "string", "enum": list(ACTIVITIES)},
                    "confidence": {"type": "number"},
                },
                # No "state": activities.py maps activity -> state, and fewer output tokens means lower latency.
                "required": ["id", "activity", "confidence"],
            },
        }
    },
    "required": ["workers"],
}


def system_prompt() -> str:
    return SYSTEM_PROMPT.format(activities=ACTIVITY_LINES)


def user_text(worker_ids: list[int]) -> str:
    ids = ", ".join(str(i) for i in worker_ids)
    return f"Workers to classify: {ids}. Return one entry for each of these ids and no others."
