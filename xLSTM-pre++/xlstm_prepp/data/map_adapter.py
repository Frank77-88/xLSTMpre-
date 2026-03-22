"""Generic lightweight map adapter for TopologyLite."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import yaml

OBSTACLE_KEYS = [
    "OBSTACLES",
    "STATIC_OBSTACLES",
    "BOUNDARY_OBSTACLES",
    "NO_GO_AREAS",
    "MEDIANS",
    "ISLANDS",
    "BLOCKERS",
    "CURBS",
]


@dataclass
class MapTokenBank:
    map_meta: torch.Tensor
    slot_polygon_vertices: torch.Tensor
    slot_polygon_vertex_mask: torch.Tensor
    slot_polygon_mask: torch.Tensor
    hard_polygon_vertices: torch.Tensor
    hard_polygon_vertex_mask: torch.Tensor
    hard_polygon_mask: torch.Tensor
    waypoint_segments: torch.Tensor
    waypoint_segment_mask: torch.Tensor
    hard_segments: torch.Tensor
    hard_segment_mask: torch.Tensor


class MapAdapter:
    """Parse parking-map style YAML into generic hard geometry banks.

    The adapter intentionally keeps only the fields required by the lightweight
    TopologyLite model:
    - hard polygons: parking areas + explicit obstacle polygons
    - waypoint segments: undirected local topology skeleton segments
    - hard segments: polygon edges + explicit segments + optional slot dividers + map bounds

    It remains map-file driven, so later replacing the YAML with a ROS-exported
    map only requires providing equivalent polygon / waypoint collections.
    """

    def __init__(
        self,
        map_path: str,
        slot_divider_as_obstacle: bool = True,
        max_slot_divider_segments: int | None = 160,
    ):
        self.map_path = str(Path(map_path).resolve())
        with open(self.map_path, "r") as file_obj:
            self.map_data = yaml.safe_load(file_obj)

        map_size = self.map_data.get("MAP_SIZE", {"x": 140.0, "y": 80.0})
        self.map_x = float(map_size.get("x", 140.0))
        self.map_y = float(map_size.get("y", 80.0))
        self.max_dim = float(max(self.map_x, self.map_y))
        self.slot_divider_as_obstacle = slot_divider_as_obstacle
        self.max_slot_divider_segments = max_slot_divider_segments

        self.parking_polygons: List[List[Tuple[float, float]]] = []
        self.obstacle_polygons: List[List[Tuple[float, float]]] = []
        self.waypoint_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self.slot_divider_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self.explicit_obstacle_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self.hard_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

        self._build_geometry()

    @staticmethod
    def _as_point(value) -> Tuple[float, float]:
        return float(value[0]), float(value[1])

    def _extract_points(self, item) -> List[Tuple[float, float]]:
        if item is None:
            return []
        if isinstance(item, dict):
            bounds = item.get("bounds")
            if bounds is None:
                return []
            return [self._as_point(point) for point in bounds]
        if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], (list, tuple)):
            return [self._as_point(point) for point in item]
        return []

    def _order_polygon(self, points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(points) <= 2:
            return list(points)
        center_x = sum(point[0] for point in points) / len(points)
        center_y = sum(point[1] for point in points) / len(points)
        return [
            (float(x), float(y))
            for x, y in sorted(points, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
        ]

    def _iter_edges(self, points: Sequence[Tuple[float, float]]) -> Iterable[Tuple[Tuple[float, float], Tuple[float, float]]]:
        for index in range(len(points)):
            yield points[index], points[(index + 1) % len(points)]

    def _polyline_to_segments(self, points: Sequence[Tuple[float, float]]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if len(points) <= 1:
            return []
        return [(points[index], points[index + 1]) for index in range(len(points) - 1)]

    @staticmethod
    def _lerp(start: Tuple[float, float], end: Tuple[float, float], ratio: float) -> Tuple[float, float]:
        return (start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio)

    def _sample_list(self, values: Sequence, max_items: int) -> List:
        if max_items <= 0:
            return []
        if len(values) <= max_items:
            return list(values)
        if max_items == 1:
            return [values[0]]
        indices = [round(index * (len(values) - 1) / (max_items - 1)) for index in range(max_items)]
        return [values[index] for index in indices]

    def _derive_slot_dividers(
        self,
        polygon: Sequence[Tuple[float, float]],
        area_specs: Sequence[dict],
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if not self.slot_divider_as_obstacle or len(polygon) != 4 or not area_specs:
            return []
        dividers: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        p0, p1, p2, p3 = polygon
        for area_spec in area_specs:
            shape = area_spec.get("shape") if isinstance(area_spec, dict) else None
            if not shape or len(shape) != 2:
                continue
            rows = max(int(shape[0]), 1)
            cols = max(int(shape[1]), 1)
            for row_idx in range(1, rows):
                ratio = row_idx / rows
                dividers.append((self._lerp(p0, p3, ratio), self._lerp(p1, p2, ratio)))
            for col_idx in range(1, cols):
                ratio = col_idx / cols
                dividers.append((self._lerp(p0, p1, ratio), self._lerp(p3, p2, ratio)))
        if self.max_slot_divider_segments is not None and len(dividers) > self.max_slot_divider_segments:
            dividers = self._sample_list(dividers, self.max_slot_divider_segments)
        return dividers

    def _parse_waypoints(self) -> None:
        for waypoint in self.map_data.get("WAYPOINTS", {}).values():
            points = self._extract_points(waypoint)
            if len(points) < 2:
                continue
            self.waypoint_segments.extend(self._polyline_to_segments(points))

    def _parse_parking_areas(self) -> None:
        for area in self.map_data.get("PARKING_AREAS", {}).values():
            points = self._extract_points(area)
            if len(points) < 3:
                continue
            polygon = self._order_polygon(points)
            self.parking_polygons.append(polygon)
            self.hard_segments.extend(list(self._iter_edges(polygon)))
            self.slot_divider_segments.extend(self._derive_slot_dividers(polygon, area.get("areas", [])))

    def _parse_collection(self, collection, close_polygon: bool) -> Tuple[List[List[Tuple[float, float]]], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        polygons: List[List[Tuple[float, float]]] = []
        segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        items = collection.values() if isinstance(collection, dict) else collection
        for item in items:
            points = self._extract_points(item)
            if len(points) >= 3 and close_polygon:
                polygons.append(self._order_polygon(points))
            elif len(points) >= 2:
                segments.extend(self._polyline_to_segments(points))
        return polygons, segments

    def _parse_explicit_obstacles(self) -> None:
        for key in OBSTACLE_KEYS:
            collection = self.map_data.get(key)
            if collection is None:
                continue
            polygons, segments = self._parse_collection(collection, close_polygon=True)
            self.obstacle_polygons.extend(polygons)
            self.explicit_obstacle_segments.extend(segments)

    def _build_geometry(self) -> None:
        self._parse_parking_areas()
        self._parse_waypoints()
        self._parse_explicit_obstacles()

        for polygon in self.obstacle_polygons:
            self.hard_segments.extend(list(self._iter_edges(polygon)))
        self.hard_segments.extend(self.explicit_obstacle_segments)
        self.hard_segments.extend(self.slot_divider_segments)
        self.hard_segments.extend(
            [
                ((0.0, 0.0), (self.map_x, 0.0)),
                ((self.map_x, 0.0), (self.map_x, self.map_y)),
                ((self.map_x, self.map_y), (0.0, self.map_y)),
                ((0.0, self.map_y), (0.0, 0.0)),
            ]
        )

    @staticmethod
    def _pack_polygons(polygons: Sequence[Sequence[Tuple[float, float]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not polygons:
            return (
                torch.zeros(0, 0, 2, dtype=torch.float32),
                torch.zeros(0, 0, dtype=torch.bool),
                torch.zeros(0, dtype=torch.bool),
            )
        max_vertices = max(len(polygon) for polygon in polygons)
        vertices = torch.zeros(len(polygons), max_vertices, 2, dtype=torch.float32)
        vertex_mask = torch.zeros(len(polygons), max_vertices, dtype=torch.bool)
        polygon_mask = torch.zeros(len(polygons), dtype=torch.bool)
        for polygon_index, polygon in enumerate(polygons):
            polygon_mask[polygon_index] = True
            for vertex_index, point in enumerate(polygon):
                vertices[polygon_index, vertex_index] = torch.tensor(point, dtype=torch.float32)
                vertex_mask[polygon_index, vertex_index] = True
        return vertices, vertex_mask, polygon_mask

    @staticmethod
    def _pack_segments(segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not segments:
            return torch.zeros(0, 2, 2, dtype=torch.float32), torch.zeros(0, dtype=torch.bool)
        segment_tensor = torch.zeros(len(segments), 2, 2, dtype=torch.float32)
        segment_mask = torch.ones(len(segments), dtype=torch.bool)
        for index, segment in enumerate(segments):
            segment_tensor[index, 0] = torch.tensor(segment[0], dtype=torch.float32)
            segment_tensor[index, 1] = torch.tensor(segment[1], dtype=torch.float32)
        return segment_tensor, segment_mask

    def build_token_bank(self) -> MapTokenBank:
        hard_polygons = self.parking_polygons + self.obstacle_polygons
        slot_polygon_vertices, slot_polygon_vertex_mask, slot_polygon_mask = self._pack_polygons(self.parking_polygons)
        hard_polygon_vertices, hard_polygon_vertex_mask, hard_polygon_mask = self._pack_polygons(hard_polygons)
        waypoint_segments, waypoint_segment_mask = self._pack_segments(self.waypoint_segments)
        hard_segments, hard_segment_mask = self._pack_segments(self.hard_segments)
        map_meta = torch.tensor([self.map_x, self.map_y, self.max_dim], dtype=torch.float32)
        return MapTokenBank(
            map_meta=map_meta,
            slot_polygon_vertices=slot_polygon_vertices,
            slot_polygon_vertex_mask=slot_polygon_vertex_mask,
            slot_polygon_mask=slot_polygon_mask,
            hard_polygon_vertices=hard_polygon_vertices,
            hard_polygon_vertex_mask=hard_polygon_vertex_mask,
            hard_polygon_mask=hard_polygon_mask,
            waypoint_segments=waypoint_segments,
            waypoint_segment_mask=waypoint_segment_mask,
            hard_segments=hard_segments,
            hard_segment_mask=hard_segment_mask,
        )
