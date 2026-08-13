from __future__ import annotations

import heapq
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple


class TrackStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    DELAYED = "delayed"


@dataclass
class RailSegment:
    id: str
    from_station: str
    to_station: str
    distance_km: float
    max_speed_kmh: float
    capacity_per_hour: int = 1
    status: TrackStatus = TrackStatus.OPEN
    delay_min: float = 0.0
    bidirectional: bool = True

    def base_travel_time_min(self) -> float:
        if self.max_speed_kmh <= 0:
            return float("inf")
        return self.distance_km / self.max_speed_kmh * 60.0

    def effective_travel_time_min(self) -> float:
        if self.status == TrackStatus.CLOSED:
            return float("inf")
        return self.base_travel_time_min() + max(0.0, self.delay_min)

    def set_delay(self, delay_min: float) -> None:
        self.delay_min = max(0.0, float(delay_min))
        self.status = TrackStatus.DELAYED if self.delay_min > 0 else TrackStatus.OPEN

    def block(self) -> None:
        self.status = TrackStatus.CLOSED

    def recover(self) -> None:
        self.status = TrackStatus.OPEN
        self.delay_min = 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data


class RailNetwork:
    """다익스트라 경로 탐색 + 시간대별 선로 slot 예약."""

    def __init__(self) -> None:
        self._segments: Dict[str, RailSegment] = {}
        self._adjacency: Dict[str, List[Tuple[str, str]]] = {}
        self._slot_reservations: Dict[Tuple[str, str], int] = {}

    def add_segment(self, segment: RailSegment) -> None:
        self._segments[segment.id] = segment
        self._adjacency.setdefault(segment.from_station, []).append((segment.to_station, segment.id))
        if segment.bidirectional:
            self._adjacency.setdefault(segment.to_station, []).append((segment.from_station, segment.id))

    def get_segment(self, segment_id: str) -> RailSegment:
        return self._segments[segment_id]

    def all_segments(self) -> List[RailSegment]:
        return list(self._segments.values())

    def segment_between(self, station_a: str, station_b: str) -> Optional[RailSegment]:
        for other, seg_id in self._adjacency.get(station_a, []):
            if other == station_b:
                return self._segments[seg_id]
        return None

    def find_route(self, origin: str, destination: str) -> Optional[Tuple[List[str], List[str]]]:
        if origin == destination:
            return [origin], []
        dist: Dict[str, float] = {origin: 0.0}
        prev: Dict[str, Tuple[str, str]] = {}
        heap: List[Tuple[float, str]] = [(0.0, origin)]
        visited = set()

        while heap:
            d, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            if node == destination:
                break
            for neighbor, seg_id in self._adjacency.get(node, []):
                seg = self._segments[seg_id]
                weight = seg.effective_travel_time_min()
                if weight == float("inf"):
                    continue
                nd = d + weight
                if nd < dist.get(neighbor, float("inf")):
                    dist[neighbor] = nd
                    prev[neighbor] = (node, seg_id)
                    heapq.heappush(heap, (nd, neighbor))

        if destination not in dist:
            return None

        stations = [destination]
        segments: List[str] = []
        node = destination
        while node != origin:
            pnode, seg_id = prev[node]
            stations.append(pnode)
            segments.append(seg_id)
            node = pnode
        stations.reverse()
        segments.reverse()
        return stations, segments

    def travel_time_min(self, origin: str, destination: str) -> Optional[float]:
        route = self.find_route(origin, destination)
        if route is None:
            return None
        _, segments = route
        return sum(self._segments[sid].effective_travel_time_min() for sid in segments)

    @staticmethod
    def time_slot(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:00")

    def is_slot_available(self, segment_id: str, dt: datetime) -> bool:
        seg = self._segments[segment_id]
        key = (segment_id, self.time_slot(dt))
        used = self._slot_reservations.get(key, 0)
        return seg.status != TrackStatus.CLOSED and used < seg.capacity_per_hour

    def can_reserve_route(self, origin: str, destination: str, departure_time: datetime) -> bool:
        route = self.find_route(origin, destination)
        if route is None:
            return False
        _, segments = route
        cursor = departure_time
        for sid in segments:
            if not self.is_slot_available(sid, cursor):
                return False
            cursor += timedelta(minutes=self._segments[sid].effective_travel_time_min())
        return True

    def reserve_route(self, origin: str, destination: str, departure_time: datetime) -> Tuple[List[str], float]:
        route = self.find_route(origin, destination)
        if route is None:
            raise RuntimeError(f"No available route: {origin}->{destination}")
        _, segments = route
        cursor = departure_time
        reserved_keys: List[Tuple[str, str]] = []
        total = 0.0
        for sid in segments:
            if not self.is_slot_available(sid, cursor):
                for key in reserved_keys:
                    self._slot_reservations[key] -= 1
                raise RuntimeError(f"Rail slot unavailable for {sid} at {cursor.isoformat()}")
            key = (sid, self.time_slot(cursor))
            self._slot_reservations[key] = self._slot_reservations.get(key, 0) + 1
            reserved_keys.append(key)
            travel = self._segments[sid].effective_travel_time_min()
            total += travel
            cursor += timedelta(minutes=travel)
        return segments, total

    def next_available_departure(
        self,
        origin: str,
        destination: str,
        earliest: datetime,
        search_hours: int = 8,
        increment_min: int = 15,
    ) -> datetime:
        cursor = earliest
        end = earliest + timedelta(hours=search_hours)
        while cursor <= end:
            if self.can_reserve_route(origin, destination, cursor):
                return cursor
            cursor += timedelta(minutes=increment_min)
        raise RuntimeError(f"No rail slot within {search_hours}h: {origin}->{destination}")

    def route_delay_min(self, segment_ids: List[str]) -> float:
        return sum(max(0.0, self._segments[sid].delay_min) for sid in segment_ids)

    def to_dict(self) -> dict:
        return {
            "segments": {sid: seg.to_dict() for sid, seg in self._segments.items()},
            "slot_reservations": {f"{sid}|{slot}": count for (sid, slot), count in self._slot_reservations.items()},
        }
