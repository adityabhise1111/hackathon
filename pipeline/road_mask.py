"""
Road-area segmentation.

Why this is cheap despite SAM being a heavy model: the drone HOVERS. Measured on
this footage, across 6824 telemetry records, relative altitude varies by 0.023 m,
gimbal pitch by 0.0 deg and yaw by 0.4 deg. The scene is therefore static in image
space, so the road mask is computed from a SINGLE frame and reused for every frame
of the clip. A one-off 40 MB MobileSAM pass costs seconds, not hours.

Two independent estimates are produced and then combined:

1. `sam_mask`        - geometric. MobileSAM segments the frame; we keep the
                       components that the traffic actually travels through.
2. `travelled_mask`  - empirical. Rasterise every trajectory ground-contact point
                       we observed and dilate by a lane width. This is where road
                       users demonstrably drove.

Neither alone is trustworthy. SAM does not know which of its segments is road, and
the travelled mask only covers road that traffic already used. Their AGREEMENT is a
high-confidence drivable area, and their disagreement is itself informative
(SAM-only = road nobody used this minute; travelled-only = SAM under-segmented).

The mask is reported with a measured agreement score rather than asserted correct.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd


def _largest_components(mask: np.ndarray, min_area_frac: float = 0.005) -> np.ndarray:
    """Drop specks; keep components covering at least `min_area_frac` of the frame."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    keep = np.zeros_like(mask, dtype=np.uint8)
    min_area = min_area_frac * mask.size
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255
    return keep


def travelled_mask(
    obs: pd.DataFrame,
    shape: tuple[int, int],
    lane_width_px: int = 26,
) -> np.ndarray:
    """
    Empirical drivable area: where road users were actually observed.

    Uses the bbox bottom-centre (the road-contact point), dilated by roughly a
    lane half-width, then closed to bridge gaps between lanes.
    """
    h, w = shape
    canvas = np.zeros((h, w), np.uint8)
    pts = obs.dropna(subset=["x", "y"])[["x", "y"]].to_numpy()
    if len(pts) == 0:
        return canvas

    xi = np.clip(pts[:, 0].astype(np.int32), 0, w - 1)
    yi = np.clip(pts[:, 1].astype(np.int32), 0, h - 1)
    canvas[yi, xi] = 255

    k = max(3, int(lane_width_px) | 1)
    canvas = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # Close across lane gaps, then fill enclosed holes (traffic islands excepted).
    big = max(3, int(lane_width_px * 2) | 1)
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (big, big)))
    return _largest_components(canvas)


def sam_road_mask(
    frame: np.ndarray,
    seed_points: np.ndarray,
    weights: str = "mobile_sam.pt",
    max_seeds: int = 24,
) -> tuple[np.ndarray | None, str]:
    """
    Segment the road with MobileSAM, prompted by points where traffic was observed.

    Prompting matters: SAM is class-agnostic, so an unprompted call returns every
    object in the scene with no idea which is road. Seeding it with real trajectory
    points means the returned segments are the surfaces traffic travels on.

    Returns (mask, note). mask is None if the model is unavailable - the caller then
    falls back to the empirical mask rather than failing.
    """
    try:
        from ultralytics import SAM
    except Exception as exc:                       # pragma: no cover
        return None, f"ultralytics SAM unavailable: {exc}"

    if len(seed_points) == 0:
        return None, "no trajectory seed points available"

    # Spread the seeds over the travelled area instead of clustering them all in
    # the busiest spot, so distinct road arms each get prompted.
    idx = np.linspace(0, len(seed_points) - 1, min(max_seeds, len(seed_points))).astype(int)
    seeds = seed_points[idx]

    try:
        model = SAM(weights)
        res = model(frame, points=seeds.tolist(), labels=[1] * len(seeds), verbose=False)
    except Exception as exc:
        return None, f"SAM inference failed: {exc}"

    h, w = frame.shape[:2]
    acc = np.zeros((h, w), np.uint8)
    got = 0
    for r in res:
        if r.masks is None:
            continue
        for m in r.masks.data.cpu().numpy():
            mm = (m > 0.5).astype(np.uint8) * 255
            if mm.shape != (h, w):
                mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
            # A single SAM segment covering almost the whole frame is a failure
            # mode (it grabbed the entire image), not a road.
            if mm.mean() / 255.0 > 0.85:
                continue
            acc = np.maximum(acc, mm)
            got += 1

    if got == 0:
        return None, "SAM returned no usable masks"
    return _largest_components(acc), f"MobileSAM, {len(seeds)} trajectory-seeded prompts, {got} segments"


def build_road_mask(
    obs: pd.DataFrame,
    frame: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    """
    Combine the empirical and geometric estimates into one drivable-area mask.

    Returns (mask_uint8, info). `info` carries the measured agreement between the
    two independent estimates, so the mask's quality is reported rather than assumed.
    """
    scfg = cfg.get("segmentation", {}) or {}
    h, w = frame.shape[:2]

    lane_px = int(scfg.get("lane_width_px", 26))
    trav = travelled_mask(obs, (h, w), lane_width_px=lane_px)
    trav_frac = float((trav > 0).mean())

    info = {
        "enabled": True,
        "static_scene_justification": (
            "drone hovers (altitude spread 0.023 m, pitch 0.0 deg, yaw 0.4 deg over "
            "6824 telemetry records), so one mask serves every frame"
        ),
        "travelled_area_frac": round(trav_frac, 4),
        "sam": None,
        "agreement": None,
        "source": "travelled_only",
    }

    sam = None
    if scfg.get("use_sam", True):
        seeds = obs.dropna(subset=["x", "y"])[["x", "y"]].to_numpy()
        sam, note = sam_road_mask(
            frame, seeds, weights=scfg.get("sam_weights", "mobile_sam.pt"),
            max_seeds=int(scfg.get("sam_max_seeds", 24)),
        )
        info["sam"] = note

    if sam is None:
        # Honest degradation: empirical mask only, and say so.
        return trav, info

    inter = cv2.bitwise_and(trav, sam)
    union = cv2.bitwise_or(trav, sam)
    iou = float((inter > 0).sum() / max((union > 0).sum(), 1))
    # How much of the travelled area SAM also called road. This is the number that
    # matters: if SAM missed road that traffic demonstrably used, it under-segmented.
    recall = float((inter > 0).sum() / max((trav > 0).sum(), 1))
    info["agreement"] = {
        "iou": round(iou, 3),
        "sam_covers_travelled_frac": round(recall, 3),
        "sam_area_frac": round(float((sam > 0).mean()), 4),
    }

    # Union, not intersection: we want road SAM found that was simply unused this
    # minute (empty lanes, far arms) included, while the travelled mask guarantees
    # we never lose road we have direct evidence for.
    combined = _largest_components(cv2.bitwise_or(trav, sam))
    info["source"] = "sam_union_travelled"
    info["road_area_frac"] = round(float((combined > 0).mean()), 4)
    return combined, info


def mask_ground_area_m2(mask: np.ndarray, projector) -> float | None:
    """
    Physical area of the mask in square metres, via the ground projection.

    Sampled on a coarse grid: each sampled pixel contributes its own local
    metres-per-pixel scale squared, because an oblique camera means scale varies
    across the frame (this one is at -63 deg pitch, not nadir).
    """
    if projector is None:
        return None
    h, w = mask.shape[:2]
    step = max(8, min(h, w) // 120)
    total = 0.0
    for v in range(0, h, step):
        for u in range(0, w, step):
            if mask[v, u] == 0:
                continue
            s = projector.ground_scale_at(u, v)
            if s is None or not np.isfinite(s):
                continue
            total += (s * step) ** 2
    return round(total, 1)


def filter_to_road(obs: pd.DataFrame, mask: np.ndarray, classes_exempt=("pedestrian",)) -> tuple[pd.DataFrame, dict]:
    """
    Drop observations whose road-contact point falls outside the drivable area.

    Pedestrians are exempt by default: footpaths are legitimately off-carriageway,
    so masking them out would delete real vulnerable road users - exactly the
    opposite of what a safety system should do.
    """
    if obs.empty or mask is None:
        return obs, {"applied": False}

    h, w = mask.shape[:2]
    xi = np.clip(obs["x"].to_numpy(), 0, w - 1).astype(np.int32)
    yi = np.clip(obs["y"].to_numpy(), 0, h - 1).astype(np.int32)
    on_road = mask[yi, xi] > 0
    exempt = obs["class"].isin(classes_exempt).to_numpy()
    keep = on_road | exempt

    dropped_tracks = sorted(set(obs.loc[~keep, "track_id"]) - set(obs.loc[keep, "track_id"]))
    stats = {
        "applied": True,
        "observations_before": int(len(obs)),
        "observations_kept": int(keep.sum()),
        "observations_dropped": int((~keep).sum()),
        "tracks_dropped_entirely": len(dropped_tracks),
        "exempt_classes": list(classes_exempt),
        "note": "off-road detections removed; pedestrians exempt (footpaths are legitimately off-carriageway)",
    }
    return obs[keep].reset_index(drop=True), stats


def render_mask_overlay(frame: np.ndarray, mask: np.ndarray, out_path: str) -> str:
    """Save a visual proof image: the frame with the drivable area tinted."""
    over = frame.copy()
    tint = np.zeros_like(frame)
    tint[mask > 0] = (0, 150, 60)
    over = cv2.addWeighted(over, 1.0, tint, 0.35, 0)
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, (60, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(over, "drivable area (MobileSAM + observed trajectories)",
                (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 255, 120), 2, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, over)
    return out_path
