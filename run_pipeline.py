"""
Traffic Intelligence - main pipeline runner.

    drone video
        -> YOLO detection
        -> ByteTrack (stable IDs)
        -> trajectory extraction (metric, via DJI telemetry)
        -> analytics / interactions / anomalies
        -> annotated video + CSVs + summary JSON

Usage:
    python run_pipeline.py                          # uses config/config.yaml
    python run_pipeline.py --max-frames 150         # quick smoke test
    python run_pipeline.py --video data/multi_road.mp4 --srt data/multi_road_full.srt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import analytics, anomalies, interactions, visualize
from pipeline.calibration import build_projector
from pipeline.detector import load_detector
from pipeline.tracker import iter_tracked_frames, video_info
from pipeline.trajectory import (
    TrajectoryStore,
    apply_track_classes,
    compute_kinematics,
    resolve_track_classes,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Drone traffic intelligence pipeline")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--video", default=None, help="override video path")
    ap.add_argument("--srt", default=None, help="override telemetry SRT path")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--no-video", action="store_true", help="skip annotated video rendering")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument(
        "--from-trajectories", nargs="?", const="__default__", default=None,
        metavar="CSV",
        help="skip detection+tracking and re-derive all analytics from an existing "
             "trajectories CSV. This is the architectural payoff: detection is the only "
             "expensive stage, so retuning a threshold or adding a new analytic costs "
             "seconds instead of a full re-processing run.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.video:
        cfg["video"]["path"] = args.video
    if args.srt:
        cfg["telemetry"]["srt_path"] = args.srt
    if args.max_frames is not None:
        cfg["video"]["max_frames"] = args.max_frames
    if args.stride is not None:
        cfg["video"]["frame_stride"] = args.stride
    if args.no_video:
        cfg["output"]["write_video"] = False

    video_path = cfg["video"]["path"]
    stride = max(1, int(cfg["video"].get("frame_stride", 1)))
    max_frames = cfg["video"].get("max_frames")
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    def out(key: str) -> str:
        base = cfg["output"][key]
        if not args.tag:
            return base
        root, ext = os.path.splitext(base)
        return f"{root}_{args.tag}{ext}"

    vinfo = video_info(video_path)
    fps = vinfo["fps"]
    print(f"[1/6] video {video_path}  {vinfo['width']}x{vinfo['height']} @ {fps:.2f}fps  "
          f"{vinfo['frames']} frames  stride={stride}")

    # ---- Calibration -------------------------------------------------------
    projector, calib_info = build_projector(cfg, vinfo["width"], vinfo["height"])
    calibrated = projector is not None
    if calibrated:
        print(f"      calibrated from DJI telemetry: alt={calib_info['altitude_m']}m  "
              f"pitch={calib_info['gimbal_pitch_deg']}deg  "
              f"{calib_info['metres_per_px_at_centre']} m/px at centre")
    else:
        print(f"      NO metric calibration ({calib_info['reason']}) -> pixel-space analytics only")

    # ---- Detection + tracking ---------------------------------------------
    # (loaded lazily inside the else-branch below: --from-trajectories must not
    #  pay the cost of loading YOLO at all.)

    store = TrajectoryStore(projector)
    box_rows: list[dict] = []
    t0 = time.time()
    n_frames = 0

    reuse = args.from_trajectories
    if reuse:
        # Re-analysis path: detection/tracking already happened, so read the saved
        # trajectories and jump straight to analytics.
        traj_path = out("trajectories_csv") if reuse == "__default__" else reuse
        obs = pd.read_csv(traj_path)
        boxes_path = os.path.join(out_dir, f"boxes{'_' + args.tag if args.tag else ''}.csv")
        boxes = pd.read_csv(boxes_path) if os.path.exists(boxes_path) else pd.DataFrame()
        calibrated = "sx" in obs.columns and obs["sx"].notna().any()
        n_frames = int(obs["frame"].nunique())
        elapsed = 1e-6
        print(f"[2/6] REUSING {traj_path}: {len(obs)} observations, "
              f"{obs['track_id'].nunique()} tracks - no detection run")
        if boxes.empty:
            print("      (no boxes.csv -> annotated video cannot be re-rendered)")
            cfg["output"]["write_video"] = False
        print("[3/6] trajectories: reused as-is (kinematics already computed)")
    else:
        model, det_settings = load_detector(cfg)
        print(f"[2/6] detector {det_settings['weights']} on device={det_settings['device']} "
              f"imgsz={det_settings['imgsz']} half={det_settings['half']}")

        for fr in iter_tracked_frames(
            model, video_path, det_settings, cfg["tracker"]["config"],
            frame_stride=stride, max_frames=max_frames,
        ):
            store.add_frame(fr["frame"], fr["t"], fr["tracks"])
            for tr in fr["tracks"]:
                x1, y1, x2, y2 = tr["xyxy"]
                box_rows.append(
                    {"frame": fr["frame"], "track_id": tr["track_id"],
                     "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
                )
            n_frames += 1
            if n_frames % 100 == 0:
                el = time.time() - t0
                print(f"      {n_frames} frames  {n_frames/el:.1f} fps  "
                      f"{len(fr['tracks'])} tracked in frame")

        elapsed = max(time.time() - t0, 1e-6)
        print(f"      done: {n_frames} frames in {elapsed:.1f}s ({n_frames/elapsed:.1f} fps)")

        raw = store.to_frame()
        boxes = pd.DataFrame(box_rows)
        if raw.empty:
            print("no detections - nothing to analyse")
            return 1

        # ---- Trajectories --------------------------------------------------
        print("[3/6] trajectories: kinematics + per-track class resolution")
        obs = compute_kinematics(raw, fps, cfg, calibrated)
        classes = resolve_track_classes(obs, cfg, calibrated)
        obs = apply_track_classes(obs, classes)
        obs.to_csv(out("trajectories_csv"), index=False)
        # Boxes are persisted so --from-trajectories can also re-render the video.
        boxes.to_csv(os.path.join(out_dir, f"boxes{'_' + args.tag if args.tag else ''}.csv"),
                     index=False)
        print(f"      {len(obs)} observations over {obs['track_id'].nunique()} tracks "
              f"-> {out('trajectories_csv')}")

    det_settings = locals().get("det_settings", {"weights": cfg["detector"]["weights"],
                                                "imgsz": cfg["detector"]["imgsz"],
                                                "conf": cfg["detector"]["conf"],
                                                "iou": cfg["detector"]["iou"],
                                                "classes": cfg["detector"]["classes"],
                                                "device": "reused", "half": False})

    # ---- Analytics ---------------------------------------------------------
    print("[4/6] analytics")
    summary = analytics.build_track_summary(obs, cfg, calibrated)
    counts = analytics.class_counts(summary)
    queues = analytics.detect_queues(obs, cfg, calibrated)
    congestion = analytics.congestion_timeline(obs, cfg, calibrated)
    active = analytics.active_track_timeline(obs)
    flow = analytics.directional_flow(summary, calibrated)
    turns = analytics.turning_movements(summary)
    stability = analytics.id_stability_stats(obs, summary, fps, stride)
    stationary = analytics.stationary_candidates(summary, cfg)

    inter = interactions.find_interactions(obs, cfg, calibrated)
    headways = interactions.following_headways(obs, inter, calibrated)
    # Duration + population let the summary report conflict RATES, which are
    # comparable across clips of different length and demand; a raw count is not.
    clip_duration_s = float(obs["timestamp"].max() - obs["timestamp"].min()) if not obs.empty else 0.0
    inter_summary = interactions.interaction_summary(
        inter, duration_s=clip_duration_s, n_road_users=len(summary)
    )

    events = anomalies.build_events(obs, summary, queues, inter, cfg, calibrated)
    events.to_csv(out("events_csv"), index=False)

    # Side tables the dashboard reads.
    summary.to_csv(os.path.join(out_dir, f"track_summary{'_' + args.tag if args.tag else ''}.csv"), index=False)
    for name, table in (
        ("queues", queues), ("congestion", congestion), ("active_tracks", active),
        ("directional_flow", flow), ("turning_movements", turns),
        ("interactions", inter), ("headways", headways),
    ):
        p = os.path.join(out_dir, f"{name}{'_' + args.tag if args.tag else ''}.csv")
        (table if table is not None and not table.empty else pd.DataFrame()).to_csv(p, index=False)

    speed_stats = {}
    if calibrated:
        moving = obs[obs["speed_kph"] > float(cfg["analytics"]["stationary_speed_kph"])]
        if not moving.empty:
            speed_stats = {
                "unit": "km/h",
                "basis": "telemetry-calibrated ground plane; estimate",
                "mean_moving_speed": round(float(moving["speed_kph"].mean()), 1),
                "median_moving_speed": round(float(moving["speed_kph"].median()), 1),
                "p85_speed": round(float(np.nanpercentile(moving["speed_kph"], 85)), 1),
                # p99.5 is the headline "top speed": the raw max is a handful of
                # frames on 4 of 292 tracks (measured), i.e. box jitter, not traffic.
                "robust_max_speed_p99_5": round(float(np.nanpercentile(moving["speed_kph"], 99.5)), 1),
                "max_speed": round(float(moving["speed_kph"].max()), 1),
                "max_speed_caveat": "raw max is jitter-sensitive; use p85 / p99.5 for reporting",
            }
    else:
        speed_stats = {"unit": "px/s", "basis": "no metric calibration available - km/h not reported",
                       "mean_moving_speed": round(float(obs["speed_px_s"].mean()), 1)}

    summary_json = {
        "video": {**vinfo, "path": video_path, "stride": stride,
                  "frames_processed": n_frames, "processing_fps": round(n_frames / elapsed, 2)},
        "detector": det_settings,
        "tracker": {"type": "bytetrack", "config": cfg["tracker"]["config"]},
        "calibration": calib_info,
        "counts": counts,
        "id_stability": stability,
        "speed": speed_stats,
        "congestion": {
            "levels": congestion["level"].value_counts().to_dict() if not congestion.empty else {},
            "peak_score": round(float(congestion["score"].max()), 3) if not congestion.empty else None,
            "method": "weighted blend of slow-fraction, speed deficit vs observed free-flow, and relative density; analytical estimate, not a certified level of service",
        },
        "queues": {
            "detections": int(len(queues)) if not queues.empty else 0,
            "largest_vehicles": int(queues["vehicles"].max()) if not queues.empty else 0,
            "longest": float(queues["length"].max()) if not queues.empty else 0.0,
            "unit": queues["length_unit"].iloc[0] if not queues.empty else None,
        },
        "turning_movements": turns.groupby("turn_type")["road_users"].sum().to_dict() if not turns.empty else {},
        "stationary_candidates": int(len(stationary)),
        "interactions": inter_summary,
        "events": {
            "total": int(len(events)),
            "by_type": events["event_type"].value_counts().to_dict() if not events.empty else {},
            "by_severity": events["severity"].value_counts().to_dict() if not events.empty else {},
        },
    }
    with open(out("summary_json"), "w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2)

    print(f"      road users: {counts['total']}  {counts['by_class']}")
    print(f"      events: {summary_json['events']['total']}  {summary_json['events']['by_type']}")
    print(f"      interactions: {inter_summary}")

    # ---- Trajectory map ----------------------------------------------------
    print("[5/6] trajectory map")
    canvas = visualize.render_trajectory_map(obs, summary, calibrated)
    if canvas is not None:
        mp = os.path.join(out_dir, f"trajectory_map{'_' + args.tag if args.tag else ''}.png")
        cv2.imwrite(mp, canvas)
        print(f"      -> {mp}")

    # ---- Annotated video ---------------------------------------------------
    if cfg["output"].get("write_video", True):
        print("[6/6] rendering annotated video")
        r = visualize.render_annotated_video(
            video_path=video_path, out_path=out("annotated_video"),
            obs=obs, boxes=boxes, summary=summary, events=events,
            congestion=congestion, counts=counts, calibrated=calibrated,
            fps=fps, frame_stride=stride,
            trail_len=int(cfg["output"].get("trail_len", 45)),
            max_frames=max_frames,
        )
        print(f"      -> {r['output']} ({r['frames_written']} frames)")
    else:
        print("[6/6] skipped annotated video (--no-video)")

    print("\nDONE. outputs in", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
