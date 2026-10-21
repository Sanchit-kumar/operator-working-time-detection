"""Per-worker hysteresis over LLM answers so labels don't flicker between ticks."""

from dataclasses import dataclass

from .activities import IDLE, UNKNOWN, WORKING


@dataclass
class _Track:
    state: str = UNKNOWN
    candidate: str = UNKNOWN
    streak: int = 0


class Smoother:
    def __init__(self, idle_after: int = 2, work_after: int = 1):
        self.need = {IDLE: idle_after, WORKING: work_after}
        self.tracks: dict[int, _Track] = {}

    def update(self, track_id: int, raw_state: str) -> str:
        """Feed one raw answer (in time order); return the smoothed state. `unknown` keeps the current state."""
        tr = self.tracks.setdefault(track_id, _Track())
        if raw_state not in self.need:
            return tr.state
        if raw_state == tr.candidate:
            tr.streak += 1
        else:
            tr.candidate, tr.streak = raw_state, 1
        # The first answer sets the state right away; after that a change needs a streak.
        if tr.state == UNKNOWN or tr.streak >= self.need[raw_state]:
            tr.state = raw_state
        return tr.state

    def state(self, track_id: int) -> str:
        return self.tracks.get(track_id, _Track()).state
