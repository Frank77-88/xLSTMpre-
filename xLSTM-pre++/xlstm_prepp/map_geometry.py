"""Shared torch geometry utilities for local topology features and safety."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

EPS = 1e-6
BIG_DISTANCE = 1e6


def normalize_vector(vector: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    norm = torch.norm(vector, dim=-1, keepdim=True).clamp_min(eps)
    return vector / norm


def heading_vector_from_positions(positions: torch.Tensor) -> torch.Tensor:
    """Estimate heading vectors from position sequences.

    Args:
        positions: [..., T, 2]

    Returns:
        [..., T, 2] normalized heading vectors.
    """
    if positions.shape[-2] <= 1:
        heading = torch.zeros_like(positions)
        heading[..., 0] = 1.0
        return heading

    delta = positions[..., 1:, :] - positions[..., :-1, :]
    first_delta = delta[..., :1, :]
    heading = torch.cat([first_delta, delta], dim=-2)
    heading_norm = torch.norm(heading, dim=-1, keepdim=True)
    fallback = torch.zeros_like(heading)
    fallback[..., 0] = 1.0
    heading = torch.where(heading_norm > EPS, heading / heading_norm.clamp_min(EPS), fallback)
    return heading


def heading_to_vector(heading: torch.Tensor) -> torch.Tensor:
    if heading.ndim > 1 and heading.shape[-1] == 2:
        return normalize_vector(heading)
    return torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)


def build_vehicle_corners(
    position: torch.Tensor,
    heading: torch.Tensor,
    agent_size: torch.Tensor,
) -> torch.Tensor:
    """Build four vehicle corners from center position / heading / size.

    Args:
        position: [N, 2]
        heading: [N] or [N, 2]
        agent_size: [N, 2] (length, width)

    Returns:
        [N, 4, 2]
    """
    forward = heading_to_vector(heading)
    lateral = torch.stack([-forward[:, 1], forward[:, 0]], dim=-1)

    half_length = agent_size[:, 0:1] * 0.5
    half_width = agent_size[:, 1:2] * 0.5

    front = forward * half_length
    side = lateral * half_width

    corners = torch.stack(
        [
            position + front + side,
            position + front - side,
            position - front - side,
            position - front + side,
        ],
        dim=1,
    )
    return corners


def point_to_segment_distance(
    points: torch.Tensor,
    segments: torch.Tensor,
    segment_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute nearest distance / projection / direction from points to segments.

    Args:
        points: [N, 2]
        segments: [N, S, 2, 2]
        segment_mask: [N, S]

    Returns:
        min_distance: [N]
        nearest_projection: [N, 2]
        nearest_direction: [N, 2]
    """
    if segments.shape[1] == 0:
        num_points = points.shape[0]
        zeros = torch.zeros(num_points, 2, device=points.device, dtype=points.dtype)
        return (
            torch.full((num_points,), BIG_DISTANCE, device=points.device, dtype=points.dtype),
            zeros,
            zeros,
        )

    start = segments[:, :, 0, :]
    end = segments[:, :, 1, :]
    segment_vec = end - start
    segment_len_sq = (segment_vec**2).sum(dim=-1).clamp_min(EPS)

    rel = points.unsqueeze(1) - start
    t = (rel * segment_vec).sum(dim=-1) / segment_len_sq
    t = t.clamp(0.0, 1.0)
    projection = start + t.unsqueeze(-1) * segment_vec
    distance = torch.norm(points.unsqueeze(1) - projection, dim=-1)
    distance = distance.masked_fill(~segment_mask.bool(), BIG_DISTANCE)

    min_distance, min_index = distance.min(dim=1)
    gather_index = min_index.view(-1, 1, 1).expand(-1, 1, 2)
    nearest_projection = projection.gather(1, gather_index).squeeze(1)

    direction = normalize_vector(segment_vec)
    nearest_direction = direction.gather(1, gather_index).squeeze(1)
    return min_distance, nearest_projection, nearest_direction


def _polygon_edges(
    polygon_vertices: torch.Tensor,
    polygon_vertex_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start = polygon_vertices
    end = torch.roll(polygon_vertices, shifts=-1, dims=2)
    edge_mask = polygon_vertex_mask.bool() & torch.roll(polygon_vertex_mask.bool(), shifts=-1, dims=2)
    return start, end, edge_mask


def min_signed_distance_to_polygons(
    points: torch.Tensor,
    polygon_vertices: torch.Tensor,
    polygon_vertex_mask: torch.Tensor,
    polygon_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Signed distance to the nearest polygon.

    Positive = outside polygons, negative = inside polygons.
    Args:
        points: [N, 2]
        polygon_vertices: [N, P, V, 2]
        polygon_vertex_mask: [N, P, V]
        polygon_mask: [N, P]

    Returns:
        min_signed_distance: [N]
        nearest_projection: [N, 2]
        inside_any: [N]
    """
    if polygon_vertices.shape[1] == 0:
        zeros = torch.zeros(points.shape[0], 2, device=points.device, dtype=points.dtype)
        return (
            torch.full((points.shape[0],), BIG_DISTANCE, device=points.device, dtype=points.dtype),
            zeros,
            torch.zeros(points.shape[0], device=points.device, dtype=torch.bool),
        )

    start, end, edge_mask = _polygon_edges(polygon_vertices, polygon_vertex_mask)
    segment_vec = end - start
    segment_len_sq = (segment_vec**2).sum(dim=-1).clamp_min(EPS)

    rel = points[:, None, None, :] - start
    t = (rel * segment_vec).sum(dim=-1) / segment_len_sq
    t = t.clamp(0.0, 1.0)
    projection = start + t.unsqueeze(-1) * segment_vec
    edge_distance = torch.norm(points[:, None, None, :] - projection, dim=-1)
    edge_distance = edge_distance.masked_fill(~edge_mask, BIG_DISTANCE)

    min_edge_distance, min_edge_index = edge_distance.min(dim=-1)
    gather_index = min_edge_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
    nearest_projection_per_polygon = projection.gather(2, gather_index).squeeze(2)

    px = points[:, None, None, 0]
    py = points[:, None, None, 1]
    xi = start[..., 0]
    yi = start[..., 1]
    xj = end[..., 0]
    yj = end[..., 1]
    crossing = ((yi > py) != (yj > py)) & (px < ((xj - xi) * (py - yi) / (yj - yi + EPS) + xi)) & edge_mask
    inside = (crossing.sum(dim=-1) % 2 == 1) & polygon_mask.bool()

    signed_distance = torch.where(inside, -min_edge_distance, min_edge_distance)
    signed_distance = signed_distance.masked_fill(~polygon_mask.bool(), BIG_DISTANCE)
    min_signed_distance, min_polygon_index = signed_distance.min(dim=1)
    gather_polygon_index = min_polygon_index.view(-1, 1, 1).expand(-1, 1, 2)
    nearest_projection = nearest_projection_per_polygon.gather(1, gather_polygon_index).squeeze(1)
    inside_any = inside.any(dim=1)
    return min_signed_distance, nearest_projection, inside_any


def map_box_signed_distance(points: torch.Tensor, map_meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Signed distance to map box.

    Positive = inside map, negative = outside map.
    Returns nearest corrective direction pointing to the safe side.
    """
    map_x = map_meta[:, 0]
    map_y = map_meta[:, 1]

    left = points[:, 0]
    right = map_x - points[:, 0]
    bottom = points[:, 1]
    top = map_y - points[:, 1]
    margins = torch.stack([left, right, bottom, top], dim=-1)

    inside = (margins >= 0.0).all(dim=-1)
    min_inside, min_inside_index = margins.min(dim=-1)
    outside_violation = (-margins).clamp_min(0.0)
    max_outside, max_outside_index = outside_violation.max(dim=-1)
    signed_distance = torch.where(inside, min_inside, -max_outside)

    direction = torch.zeros_like(points)
    inside_dirs = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        device=points.device,
        dtype=points.dtype,
    )
    direction[inside] = inside_dirs[min_inside_index[inside]]
    direction[~inside] = inside_dirs[max_outside_index[~inside]]
    return signed_distance, direction


def _repeat_geometry(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    return tensor.unsqueeze(1).expand(-1, repeats, *tensor.shape[1:]).reshape(-1, *tensor.shape[1:])


def _segment_distance_matrix(points: torch.Tensor, segments: torch.Tensor, segment_mask: torch.Tensor) -> torch.Tensor:
    if segments.shape[1] == 0:
        return torch.full((points.shape[0], 0), BIG_DISTANCE, device=points.device, dtype=points.dtype)
    start = segments[:, :, 0, :]
    end = segments[:, :, 1, :]
    segment_vec = end - start
    segment_len_sq = (segment_vec ** 2).sum(dim=-1).clamp_min(EPS)
    rel = points.unsqueeze(1) - start
    t = (rel * segment_vec).sum(dim=-1) / segment_len_sq
    t = t.clamp(0.0, 1.0)
    projection = start + t.unsqueeze(-1) * segment_vec
    distance = torch.norm(points.unsqueeze(1) - projection, dim=-1)
    return distance.masked_fill(~segment_mask.bool(), BIG_DISTANCE)


def _masked_topk_indices(distance: torch.Tensor, mask: torch.Tensor, topk: int):
    if distance.shape[1] == 0 or topk <= 0:
        empty_idx = torch.zeros(distance.shape[0], 0, dtype=torch.long, device=distance.device)
        empty_mask = torch.zeros(distance.shape[0], 0, dtype=torch.bool, device=distance.device)
        empty_val = torch.zeros(distance.shape[0], 0, dtype=distance.dtype, device=distance.device)
        return empty_idx, empty_mask, empty_val
    topk = min(int(topk), distance.shape[1])
    values, indices = torch.topk(distance, k=topk, dim=1, largest=False)
    valid = mask.gather(1, indices) & (values < BIG_DISTANCE * 0.5)
    return indices, valid, values


def _gather_segments(segments: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.shape[1] == 0:
        return torch.zeros(segments.shape[0], 0, 2, 2, device=segments.device, dtype=segments.dtype)
    gather_index = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2)
    return segments.gather(1, gather_index)


def select_local_waypoint_subset(
    position: torch.Tensor,
    waypoint_segments: torch.Tensor,
    waypoint_segment_mask: torch.Tensor,
    topk_waypoints: int = 12,
    local_radius: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    waypoint_distance = _segment_distance_matrix(position, waypoint_segments, waypoint_segment_mask)
    if local_radius is not None and local_radius > 0.0:
        radius_mask = waypoint_distance <= float(local_radius)
        waypoint_segment_mask = waypoint_segment_mask.bool() & radius_mask
        waypoint_distance = waypoint_distance.masked_fill(~waypoint_segment_mask, BIG_DISTANCE)
    waypoint_idx, waypoint_local_mask, waypoint_topv = _masked_topk_indices(
        waypoint_distance,
        waypoint_segment_mask.bool(),
        topk_waypoints,
    )
    local_waypoint_segments = _gather_segments(waypoint_segments, waypoint_idx)
    return {
        "waypoint_segments": local_waypoint_segments,
        "waypoint_mask": waypoint_local_mask,
        "waypoint_distance": waypoint_topv,
    }


def _polygon_centers_and_sizes(
    polygon_vertices: torch.Tensor,
    polygon_vertex_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if polygon_vertices.shape[1] == 0:
        zeros = torch.zeros(polygon_vertices.shape[0], 0, 2, device=polygon_vertices.device, dtype=polygon_vertices.dtype)
        empty = torch.zeros(polygon_vertices.shape[0], 0, device=polygon_vertices.device, dtype=polygon_vertices.dtype)
        return zeros, empty, empty

    mask = polygon_vertex_mask.unsqueeze(-1).to(dtype=polygon_vertices.dtype)
    denom = mask.sum(dim=2).clamp_min(1.0)
    centers = (polygon_vertices * mask).sum(dim=2) / denom
    large_pos = torch.full_like(polygon_vertices, BIG_DISTANCE)
    large_neg = torch.full_like(polygon_vertices, -BIG_DISTANCE)
    masked_min = torch.where(polygon_vertex_mask.unsqueeze(-1), polygon_vertices, large_pos).min(dim=2).values
    masked_max = torch.where(polygon_vertex_mask.unsqueeze(-1), polygon_vertices, large_neg).max(dim=2).values
    size = (masked_max - masked_min).clamp_min(0.0)
    return centers, size[:, :, 0], size[:, :, 1]


def _gather_polygon_vertices(vertices: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.shape[1] == 0:
        return torch.zeros(vertices.shape[0], 0, vertices.shape[2], 2, device=vertices.device, dtype=vertices.dtype)
    gather_index = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, vertices.shape[2], 2)
    return vertices.gather(1, gather_index)


def _gather_polygon_mask(mask: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.shape[1] == 0:
        return torch.zeros(mask.shape[0], 0, device=mask.device, dtype=mask.dtype)
    return mask.gather(1, indices)


def _gather_polygon_vertex_mask(mask: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.shape[1] == 0:
        return torch.zeros(mask.shape[0], 0, mask.shape[-1], device=mask.device, dtype=mask.dtype)
    gather_index = indices.unsqueeze(-1).expand(-1, -1, mask.shape[-1])
    return mask.gather(1, gather_index)


def select_local_map_subset(
    position: torch.Tensor,
    hard_polygon_vertices: torch.Tensor,
    hard_polygon_vertex_mask: torch.Tensor,
    hard_polygon_mask: torch.Tensor,
    waypoint_segments: torch.Tensor,
    waypoint_segment_mask: torch.Tensor,
    hard_segments: torch.Tensor,
    hard_segment_mask: torch.Tensor,
    map_meta: torch.Tensor,
    topk_waypoints: int = 8,
    topk_hard_segments: int = 8,
    topk_polygons: int = 3,
    local_radius: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    waypoint_distance = _segment_distance_matrix(position, waypoint_segments, waypoint_segment_mask)
    waypoint_valid_mask = waypoint_segment_mask.bool()
    if local_radius is not None and local_radius > 0.0:
        waypoint_valid_mask = waypoint_valid_mask & (waypoint_distance <= float(local_radius))
        waypoint_distance = waypoint_distance.masked_fill(~waypoint_valid_mask, BIG_DISTANCE)
    waypoint_idx, waypoint_local_mask, waypoint_topv = _masked_topk_indices(waypoint_distance, waypoint_valid_mask, topk_waypoints)
    local_waypoint_segments = _gather_segments(waypoint_segments, waypoint_idx)

    hard_distance = _segment_distance_matrix(position, hard_segments, hard_segment_mask)
    hard_valid_mask = hard_segment_mask.bool()
    if local_radius is not None and local_radius > 0.0:
        hard_valid_mask = hard_valid_mask & (hard_distance <= float(local_radius))
        hard_distance = hard_distance.masked_fill(~hard_valid_mask, BIG_DISTANCE)
    hard_idx, hard_local_mask, hard_topv = _masked_topk_indices(hard_distance, hard_valid_mask, topk_hard_segments)
    local_hard_segments = _gather_segments(hard_segments, hard_idx)

    poly_centers, poly_width, poly_height = _polygon_centers_and_sizes(hard_polygon_vertices, hard_polygon_vertex_mask)
    if poly_centers.shape[1] == 0:
        poly_distance = torch.zeros(position.shape[0], 0, device=position.device, dtype=position.dtype)
        poly_idx = torch.zeros(position.shape[0], 0, device=position.device, dtype=torch.long)
        poly_local_mask = torch.zeros(position.shape[0], 0, device=position.device, dtype=torch.bool)
        poly_topv = poly_distance
    else:
        poly_distance = torch.norm(poly_centers - position.unsqueeze(1), dim=-1)
        poly_valid_mask = hard_polygon_mask.bool()
        if local_radius is not None and local_radius > 0.0:
            poly_valid_mask = poly_valid_mask & (poly_distance <= float(local_radius))
        poly_distance = poly_distance.masked_fill(~poly_valid_mask, BIG_DISTANCE)
        poly_idx, poly_local_mask, poly_topv = _masked_topk_indices(poly_distance, poly_valid_mask, topk_polygons)
    local_poly_vertices = _gather_polygon_vertices(hard_polygon_vertices, poly_idx)
    local_poly_vertex_mask = _gather_polygon_vertex_mask(hard_polygon_vertex_mask, poly_idx)
    local_poly_mask = _gather_polygon_mask(hard_polygon_mask, poly_idx)
    local_poly_centers = poly_centers.gather(1, poly_idx.unsqueeze(-1).expand(-1, -1, 2)) if poly_idx.shape[1] > 0 else torch.zeros(position.shape[0], 0, 2, device=position.device, dtype=position.dtype)
    local_poly_width = poly_width.gather(1, poly_idx) if poly_idx.shape[1] > 0 else torch.zeros(position.shape[0], 0, device=position.device, dtype=position.dtype)
    local_poly_height = poly_height.gather(1, poly_idx) if poly_idx.shape[1] > 0 else torch.zeros(position.shape[0], 0, device=position.device, dtype=position.dtype)

    return {
        "waypoint_segments": local_waypoint_segments,
        "waypoint_mask": waypoint_local_mask,
        "waypoint_distance": waypoint_topv,
        "hard_segments": local_hard_segments,
        "hard_mask": hard_local_mask,
        "hard_distance": hard_topv,
        "hard_polygon_vertices": local_poly_vertices,
        "hard_polygon_vertex_mask": local_poly_vertex_mask,
        "hard_polygon_mask": local_poly_mask,
        "polygon_centers": local_poly_centers,
        "polygon_width": local_poly_width,
        "polygon_height": local_poly_height,
        "polygon_distance": poly_topv,
        "polygon_local_mask": poly_local_mask,
        "map_meta": map_meta,
    }


def build_local_topology_features(
    position: torch.Tensor,
    local_map: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    map_scale = local_map["map_meta"][:, 2:3].clamp_min(1.0)

    waypoint_segments = local_map["waypoint_segments"]
    waypoint_mask = local_map["waypoint_mask"]
    waypoint_mid = waypoint_segments.mean(dim=2) if waypoint_segments.shape[1] > 0 else torch.zeros(position.shape[0], 0, 2, device=position.device, dtype=position.dtype)
    waypoint_vec = waypoint_segments[:, :, 1, :] - waypoint_segments[:, :, 0, :] if waypoint_segments.shape[1] > 0 else waypoint_mid
    waypoint_len = torch.norm(waypoint_vec, dim=-1, keepdim=True) / map_scale.unsqueeze(1)
    waypoint_span = waypoint_vec.abs() / map_scale.unsqueeze(1)
    waypoint_rel = (waypoint_mid - position.unsqueeze(1)) / map_scale.unsqueeze(1)
    waypoint_feat = torch.cat([
        waypoint_rel,
        waypoint_rel.abs(),
        waypoint_len,
        local_map["waypoint_distance"].unsqueeze(-1) / map_scale.unsqueeze(1),
        waypoint_span,
    ], dim=-1) if waypoint_segments.shape[1] > 0 else torch.zeros(position.shape[0], 0, 8, device=position.device, dtype=position.dtype)

    hard_segments = local_map["hard_segments"]
    hard_mask = local_map["hard_mask"]
    hard_mid = hard_segments.mean(dim=2) if hard_segments.shape[1] > 0 else torch.zeros(position.shape[0], 0, 2, device=position.device, dtype=position.dtype)
    hard_vec = hard_segments[:, :, 1, :] - hard_segments[:, :, 0, :] if hard_segments.shape[1] > 0 else hard_mid
    hard_len = torch.norm(hard_vec, dim=-1, keepdim=True) / map_scale.unsqueeze(1)
    hard_span = hard_vec.abs() / map_scale.unsqueeze(1)
    hard_rel = (hard_mid - position.unsqueeze(1)) / map_scale.unsqueeze(1)
    hard_feat = torch.cat([
        hard_rel,
        hard_rel.abs(),
        hard_len,
        local_map["hard_distance"].unsqueeze(-1) / map_scale.unsqueeze(1),
        hard_span,
    ], dim=-1) if hard_segments.shape[1] > 0 else torch.zeros(position.shape[0], 0, 8, device=position.device, dtype=position.dtype)

    polygon_mask = local_map["polygon_local_mask"]
    polygon_centers = local_map["polygon_centers"]
    polygon_rel = (polygon_centers - position.unsqueeze(1)) / map_scale.unsqueeze(1) if polygon_centers.shape[1] > 0 else torch.zeros(position.shape[0], 0, 2, device=position.device, dtype=position.dtype)
    polygon_feat = torch.cat([
        polygon_rel,
        polygon_rel.abs(),
        local_map["polygon_width"].unsqueeze(-1) / map_scale.unsqueeze(1),
        local_map["polygon_height"].unsqueeze(-1) / map_scale.unsqueeze(1),
        local_map["polygon_distance"].unsqueeze(-1) / map_scale.unsqueeze(1),
        (local_map["polygon_width"] * local_map["polygon_height"]).unsqueeze(-1) / (map_scale.unsqueeze(1) ** 2),
    ], dim=-1) if polygon_centers.shape[1] > 0 else torch.zeros(position.shape[0], 0, 8, device=position.device, dtype=position.dtype)

    return {
        "waypoint_features": waypoint_feat,
        "waypoint_mask": waypoint_mask,
        "hard_features": hard_feat,
        "hard_mask": hard_mask,
        "polygon_features": polygon_feat,
        "polygon_mask": polygon_mask,
    }


def compute_point_map_bundle(
    position: torch.Tensor,
    heading: torch.Tensor,
    agent_size: torch.Tensor,
    hard_polygon_vertices: torch.Tensor,
    hard_polygon_vertex_mask: torch.Tensor,
    hard_polygon_mask: torch.Tensor,
    waypoint_segments: torch.Tensor,
    waypoint_segment_mask: torch.Tensor,
    hard_segments: torch.Tensor,
    hard_segment_mask: torch.Tensor,
    map_meta: torch.Tensor,
    safety_margin: float = 1.5,
) -> Dict[str, torch.Tensor]:
    """Compute local hard-geometry summary at the current query state."""
    heading_vec = heading_to_vector(heading)
    corners = build_vehicle_corners(position, heading_vec, agent_size)
    flat_corners = corners.reshape(-1, 2)
    repeated_map_meta = map_meta.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3)
    repeated_poly_vertices = hard_polygon_vertices.unsqueeze(1).expand(-1, 4, -1, -1, -1).reshape(
        -1, hard_polygon_vertices.shape[1], hard_polygon_vertices.shape[2], 2
    )
    repeated_poly_vertex_mask = hard_polygon_vertex_mask.unsqueeze(1).expand(-1, 4, -1, -1).reshape(
        -1, hard_polygon_vertex_mask.shape[1], hard_polygon_vertex_mask.shape[2]
    )
    repeated_poly_mask = hard_polygon_mask.unsqueeze(1).expand(-1, 4, -1).reshape(-1, hard_polygon_mask.shape[1])

    center_poly_signed, center_poly_projection, center_inside = min_signed_distance_to_polygons(
        position,
        hard_polygon_vertices,
        hard_polygon_vertex_mask,
        hard_polygon_mask,
    )
    corner_poly_signed, _, corner_inside = min_signed_distance_to_polygons(
        flat_corners,
        repeated_poly_vertices,
        repeated_poly_vertex_mask,
        repeated_poly_mask,
    )
    corner_poly_signed = corner_poly_signed.view(position.shape[0], 4)
    corner_poly_min = corner_poly_signed.min(dim=1).values
    corner_inside_ratio = corner_inside.view(position.shape[0], 4).float().mean(dim=1)

    center_box_signed, center_box_dir = map_box_signed_distance(position, map_meta)
    corner_box_signed, _ = map_box_signed_distance(flat_corners, repeated_map_meta)
    corner_box_signed = corner_box_signed.view(position.shape[0], 4)
    corner_box_min = corner_box_signed.min(dim=1).values

    lane_dist, _, _ = point_to_segment_distance(position, waypoint_segments, waypoint_segment_mask)
    hard_dist, _, _ = point_to_segment_distance(position, hard_segments, hard_segment_mask)

    safe_poly_distance = center_poly_signed.clamp_min(0.0)
    safe_box_distance = center_box_signed.clamp_min(0.0)
    poly_penetration = F.relu(-center_poly_signed) + F.relu(-corner_poly_min)
    box_penetration = F.relu(-center_box_signed) + F.relu(-corner_box_min)
    poly_margin = F.relu(safety_margin - safe_poly_distance)
    box_margin = F.relu(safety_margin - safe_box_distance)

    poly_push_vec = position - center_poly_projection
    poly_push_dir = normalize_vector(poly_push_vec)
    poly_push_sign = torch.where(center_poly_signed >= 0.0, 1.0, -1.0).unsqueeze(-1)
    poly_correction = poly_push_sign * poly_push_dir * poly_margin.unsqueeze(-1)
    box_correction = center_box_dir * box_margin.unsqueeze(-1)
    hard_correction = poly_correction + box_correction

    map_scale = map_meta[:, 2].clamp_min(1.0)
    signal = torch.stack(
        [
            center_poly_signed / map_scale,
            corner_poly_min / map_scale,
            center_box_signed / map_scale,
            corner_box_min / map_scale,
            lane_dist / map_scale,
            hard_dist / map_scale,
            poly_penetration / map_scale,
            box_penetration / map_scale,
            poly_margin / map_scale,
            box_margin / map_scale,
            corner_inside_ratio,
        ],
        dim=-1,
    )

    return {
        "signal": signal,
        "hard_correction": hard_correction,
        "poly_penetration": poly_penetration,
        "box_penetration": box_penetration,
        "poly_margin": poly_margin,
        "box_margin": box_margin,
        "center_poly_signed": center_poly_signed,
        "corner_poly_min": corner_poly_min,
        "center_box_signed": center_box_signed,
        "corner_box_min": corner_box_min,
        "lane_dist": lane_dist,
        "hard_dist": hard_dist,
        "corner_inside_ratio": corner_inside_ratio,
    }

def compute_trajectory_safety(
    trajectory: torch.Tensor,
    agent_size: torch.Tensor,
    hard_polygon_vertices: torch.Tensor,
    hard_polygon_vertex_mask: torch.Tensor,
    hard_polygon_mask: torch.Tensor,
    waypoint_segments: torch.Tensor,
    waypoint_segment_mask: torch.Tensor,
    hard_segments: torch.Tensor,
    hard_segment_mask: torch.Tensor,
    map_meta: torch.Tensor,
    safety_margin: float = 1.5,
) -> Dict[str, torch.Tensor]:
    """Compute safety risk terms for trajectories.

    Args:
        trajectory: [B, K, T, 2]
        agent_size: [B, 2]
        hard_polygon_vertices: [B, P, V, 2]
        hard_polygon_vertex_mask: [B, P, V]
        hard_polygon_mask: [B, P]
        waypoint_segments: [B, W, 2, 2]
        waypoint_segment_mask: [B, W]
        hard_segments: [B, S, 2, 2]
        hard_segment_mask: [B, S]
        map_meta: [B, 3]
    Returns:
        Dict with risk tensors [B, K].
    """
    batch_size, num_modes, pred_len, _ = trajectory.shape
    heading = heading_vector_from_positions(trajectory)

    flat_pos = trajectory.reshape(batch_size * num_modes * pred_len, 2)
    flat_heading = heading.reshape(batch_size * num_modes * pred_len, 2)
    flat_agent_size = _repeat_geometry(agent_size, num_modes * pred_len)
    flat_poly_vertices = _repeat_geometry(hard_polygon_vertices, num_modes * pred_len)
    flat_poly_vertex_mask = _repeat_geometry(hard_polygon_vertex_mask, num_modes * pred_len)
    flat_poly_mask = _repeat_geometry(hard_polygon_mask, num_modes * pred_len)
    flat_waypoint_segments = _repeat_geometry(waypoint_segments, num_modes * pred_len)
    flat_waypoint_mask = _repeat_geometry(waypoint_segment_mask, num_modes * pred_len)
    flat_hard_segments = _repeat_geometry(hard_segments, num_modes * pred_len)
    flat_hard_segment_mask = _repeat_geometry(hard_segment_mask, num_modes * pred_len)
    flat_map_meta = _repeat_geometry(map_meta, num_modes * pred_len)

    bundle = compute_point_map_bundle(
        position=flat_pos,
        heading=flat_heading,
        agent_size=flat_agent_size,
        hard_polygon_vertices=flat_poly_vertices,
        hard_polygon_vertex_mask=flat_poly_vertex_mask,
        hard_polygon_mask=flat_poly_mask,
        waypoint_segments=flat_waypoint_segments,
        waypoint_segment_mask=flat_waypoint_mask,
        hard_segments=flat_hard_segments,
        hard_segment_mask=flat_hard_segment_mask,
        map_meta=flat_map_meta,
        safety_margin=safety_margin,
    )

    poly_penetration = bundle["poly_penetration"].view(batch_size, num_modes, pred_len)
    box_penetration = bundle["box_penetration"].view(batch_size, num_modes, pred_len)
    poly_margin = bundle["poly_margin"].view(batch_size, num_modes, pred_len)
    box_margin = bundle["box_margin"].view(batch_size, num_modes, pred_len)

    intrusion_penalty = poly_penetration + box_penetration
    proximity_penalty = poly_margin + box_margin
    risk_penalty = intrusion_penalty.mean(dim=-1) + 0.25 * proximity_penalty.mean(dim=-1)
    any_intrusion = (intrusion_penalty > 0.0).any(dim=-1).float()
    return {
        "intrusion_penalty": intrusion_penalty.mean(dim=-1),
        "proximity_penalty": proximity_penalty.mean(dim=-1),
        "risk_penalty": risk_penalty,
        "intrusion_rate": any_intrusion,
    }
