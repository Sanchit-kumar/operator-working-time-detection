"""Fixed activity vocabulary the LLM chooses from, and how each maps to a Working/Idle state.

The mapping here is authoritative: the model's own `state` field is logged but not trusted.
"""

WORKING, IDLE, UNKNOWN = "working", "idle", "unknown"

ACTIVITIES: dict[str, tuple[str, str]] = {
    # name: (state, description shown to the model)
    "computer_work": (WORKING, "typing, using a computer, tablet or scanner"),
    "packing_sorting": (WORKING, "packing, sorting, assembling or handling items at a station"),
    "carrying": (WORKING, "carrying, lifting or moving goods or boxes"),
    "operating_equipment": (WORKING, "driving a forklift, pulling a pallet jack, operating a machine"),
    "walking_with_purpose": (WORKING, "walking between stations while carrying or pushing something"),
    "walking_empty_handed": (IDLE, "wandering or strolling with empty hands, no task visible"),
    "standing_idle": (IDLE, "standing still with empty hands, not engaged with any task"),
    "sitting_idle": (IDLE, "sitting without working, resting"),
    "phone": (IDLE, "using a personal phone"),
    "chatting": (IDLE, "talking with others, not working"),
    "occluded": (UNKNOWN, "mostly hidden, cannot tell"),
    "not_a_worker": (UNKNOWN, "not a person, or a visitor/passer-by"),
}

STATES = (WORKING, IDLE, UNKNOWN)


def state_of(activity: str) -> str:
    return ACTIVITIES.get(activity, (UNKNOWN, ""))[0]
