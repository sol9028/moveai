from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import heapq
import itertools
from typing import Any, Iterable


class EventType(str, Enum):
    TRACK_DELAY = "TRACK_DELAY"
    TRACK_RECOVERED = "TRACK_RECOVERED"
    TRACK_BLOCKED = "TRACK_BLOCKED"
    CURTAILMENT_UPDATED = "CURTAILMENT_UPDATED"
    DEMAND_UPDATED = "DEMAND_UPDATED"
    CARGO_READY = "CARGO_READY"
    ESS_FAILURE = "ESS_FAILURE"
    ESS_RECOVERED = "ESS_RECOVERED"
    CUSTOM = "CUSTOM"


_counter = itertools.count()


@dataclass(order=True)
class Event:
    time: datetime
    sort_index: int = field(init=False, repr=False)
    type: EventType = field(compare=False, default=EventType.CUSTOM)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.sort_index = next(_counter)


class EventQueue:
    def __init__(self, events: Iterable[Event] | None = None):
        self._heap: list[Event] = []
        if events:
            for event in events:
                self.push(event)

    def push(self, event: Event) -> None:
        heapq.heappush(self._heap, event)

    def pop_due(self, now: datetime) -> list[Event]:
        due = []
        while self._heap and self._heap[0].time <= now:
            due.append(heapq.heappop(self._heap))
        return due

    def __len__(self) -> int:
        return len(self._heap)
