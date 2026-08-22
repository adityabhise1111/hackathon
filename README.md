# Drone Traffic Intelligence - Level 1

**Detection and tracking of every road user in drone footage, with stable identities
through occlusion, and a reusable trajectory layer that all analytics are built on.**

Aerial video is not just "CCTV from higher up". A drone sees a whole intersection in
one frame, in one coordinate system - so it can measure the *relationships* between
road users, which is exactly what fixed cameras cannot do. This build leans into that.

The differentiator: **DJI aircraft embed per-frame flight telemetry in the MP4 as a
subtitle track.** Altitude, focal length and gimbal angles are enough to project any
pixel onto the road plane, so this system reports **real metres, real km/h and real
compass bearings** instead of pixel velocities.

---

## Quick start

```bash
pip install -r requirements.txt
```

Prepare a clip (downscaling 4K -> 1920 costs no metric accuracy - see ISSUES.md #1):

```bash
ffmpeg -ss 120 -i "source.MP4" -t 60 -vf scale=1920:-2 -c:v libx264 -preset ultrafast -crf 26 -an -sn data/intersection.mp4
```

Extract the telemetry (this is what unlocks metric output):

```bash
ffmpeg -i "source.MP4" -map 0:s:0 data/intersection_full.srt
```

Run the pipeline, then the dashboard:

```bash
python run_pipeline.py
```

```bash
streamlit run app.py
```

**No GPU?** Open `Traffic_Intelligence_Colab.ipynb` in Google Colab and pick a T4
runtime - the same code runs ~20x faster. It reads the video straight from Google
Drive, so there is nothing to upload.

Useful flags:

```bash
python run_pipeline.py --max-frames 60 --tag smoke      # fast smoke test
python run_pipeline.py --no-video                       # analytics only
python run_pipeline.py --from-trajectories outputs/trajectories.csv --tag retune
```

That last one is the important one - see [Architecture](#architecture) below.

---

## Architecture

```
                      drone video (MP4)
                             |
        +--------------------+--------------------+
        |                                         |
  tx3g subtitle track                        video frames
        |                                         |
  pipeline/calibration.py                 pipeline/detector.py   (YOLOv8)
  altitude, focal length,                 pipeline/tracker.py    (ByteTrack)
  gimbal angles                                   |
        |                                    stable track IDs
  pinhole -> ground plane                         |
  projection (metres)                             |
        +--------------------+--------------------+
                             |
                  pipeline/trajectory.py
              ***  trajectories.csv  ***   <-- the contract
                             |
        +----------------+---+-----------+-------------+
        |                |               |             |
   analytics.py    interactions.py   anomalies.py  visualize.py
   counts, queues,  TTC, conflicts,   events.csv   annotated.mp4
   congestion,      headways                       trajectory map
   turning, dwell
```

**`trajectories.csv` is the architectural contract.** Detection and tracking are the
only expensive stages and they run exactly once. Every analytic is a pure function of
the trajectory table - none of them import YOLO, and none can see a video frame.

Three concrete consequences:

1. **Adding an analytic for Level 2+ never touches detection or tracking.** Write a
   function that takes the trajectory DataFrame; register it in `anomalies.DETECTORS`.
2. **Retuning is free.** `--from-trajectories` re-derives every analytic in *seconds*
   instead of 25 minutes. The conflict-threshold sweep in ISSUES.md #5 and the two
   conflict fixes in #6 and #7 were all done this way, with zero GPU time.
3. **A CPU-only machine is not a blocker.** You pay detection once.

---

## Metric calibration - how real-world units are obtained

`ffprobe` on the source reveals a `tx3g` subtitle stream. Each record looks like:

```
[focal_len: 24.00] [latitude: 18.566227] [longitude: 73.771846]
[rel_alt: 70.472 abs_alt: 607.273] [gb_yaw: -125.5 gb_pitch: -63.1 gb_roll: 0.0]
```

`pipeline/calibration.py` turns that into a ground projection:

1. **Focal length in pixels** from the 35 mm-equivalent value:
   `f_px = (focal_35mm / 36.0) * image_width`
2. **Camera rotation** from gimbal yaw/pitch/roll, in an ENU world frame
   (X east, Y north, Z up), with the aircraft at `Z = rel_alt`.
3. **Ray-cast** each pixel through the pinhole model onto the plane `Z = 0`.

A ray pointing above the horizon returns `NaN`, never a fabricated coordinate.

Every road user's **bottom-centre bbox point** is used as the road-contact point,
which is where the vehicle actually meets the ground plane.

### Validating calibration without ground truth

There is no survey to check against, so we validate by **physical plausibility** -
three independent checks that would all fail on a broken projection:

| Check | Measured | Verdict |
|---|---|---|
| **Hover stability** | altitude spread 0.023 m, pitch 0.0 deg, yaw 0.4 deg across 6,824 records | Aircraft is genuinely static -> a single fixed camera pose is justified |
| **Ground scale** | 0.0617 m/px at centre, 97.6 m scene width at 70.47 m altitude | Correct order of magnitude for a 24 mm-equivalent lens at that height |
| **Speed distribution** | mean 16.7, median 16.8, p85 23.6 km/h | Exactly right for a busy urban intersection. A broken projection gives 300 km/h or 2 km/h |

**Without telemetry the system degrades honestly.** It falls back to pixel-space
analytics and **refuses to print km/h at all** rather than inventing a scale factor.

---

## Detection and tracking

**Detector:** YOLOv8s (COCO), `imgsz=1280`, `conf=0.25`, restricted to the six
relevant COCO classes - person, bicycle, car, motorcycle, bus, truck. No custom
training: within a 4-hour budget a pretrained model with tuned inference beats a
half-trained custom one.

**Tracker:** ByteTrack (bundled with Ultralytics) via `model.track(persist=True)`.
It keeps low-confidence detections as association candidates instead of discarding
them, which is precisely what carries an ID through partial occlusion.

**The tuning that mattered most** is `track_buffer: 90` (~3 s at 30 fps) in
`config/bytetrack.yaml` - long enough to survive a bus passing in front of a car, or
a vehicle waiting at a red light.

### Identity stability, measured

| Metric | 60 s clip |
|---|---|
| Road users tracked | 292 |
| Observations | 69,505 |
| **Gap recoveries** (re-linked after missed detections) | **2,688** |
| Longest bridged occlusion | 3.07 s |
| Mean track life | 9.85 s |
| Longest track | 59.96 s (the full clip) |
| Tracks over 10 s | 99 |

We have no hand-labelled ground truth, so **we do not quote MOTA or IDF1** - we
cannot compute them. Gap recoveries are the honest, directly measurable evidence
that identities survive occlusion.

---

## Trajectory representation

One row per road user per frame in `trajectories.csv`:

| column | meaning |
|---|---|
| `track_id`, `class`, `frame`, `timestamp` | identity and time |
| `x`, `y` | bottom-centre of bbox, image space (road-contact point) |
| `world_x`, `world_y` | raw ground-plane projection, metres |
| `sx`, `sy` | smoothed ground coordinates (centred moving average) |
| `speed_kph`, `speed_px_s` | centred difference over a ~0.5 s baseline |
| `bearing_deg` | true compass bearing (metric); `heading_px_deg` in pixel space |
| `accel_kph_s` | for sudden-stop detection |
| `footprint_m` | measured ground width of the bbox bottom edge |

Speed uses a **centred** difference over a ~0.5 s baseline: long enough to suppress
per-frame jitter, short enough to preserve real acceleration.

---

## Analytics

All in `pipeline/analytics.py`, `interactions.py`, `anomalies.py` - each a pure
function of the trajectory table.

**Counts and classification.** Class is resolved **per track by majority vote**, not
per frame, so detector flicker cannot double-count a vehicle.

**Stationary and dwell.** Sustained low-speed runs, cross-referenced against the
queue detector so a red light is not reported as an incident.

**Queues.** Single-linkage spatial clustering (`cKDTree` + union-find) of slow
vehicles per time bin. Length is the cluster's extent along its own principal axis
(via SVD), in metres.

**Congestion.** A transparent weighted score, stated in the output:
`0.45 * slow_fraction + 0.35 * speed_deficit + 0.20 * relative_density`.

**Turning movements.** Entry vs exit heading per trajectory. Tracks too short to
judge are excluded rather than forced into a bucket.

**Directional flow.** True 16-point compass distribution when calibrated.

**Interactions (surrogate safety).** For every pair within 25 m, solve for the
closest point of approach under constant velocity:

```
dp = p2 - p1,  dv = v2 - v1
t_cpa = -(dp . dv) / |dv|^2
d_cpa = |dp + t_cpa * dv|
```

Getting this to produce trustworthy output took **three separate fixes** - it is the
most interesting engineering in the project, and all three are documented in
[ISSUES.md](ISSUES.md) #5, #6 and #7. In short, a naive CPA implementation is wrong
in three independent ways:

1. **Speed noise** inflates `|dv|` for vehicles moving in a line -> require a real
   closing rate along the line of centres, plus multi-frame persistence.
2. **Dense-traffic norms**: a bare TTC <= 1.5 s test flags ~90% of all pairs, because
   queue creep is geometrically a "conflict" -> grade jointly on TTC **and closing
   speed**, which proxies the energy an evasive manoeuvre must absorb.
3. **Envelope geometry**: an omnidirectional radius (~half vehicle *length*, 3.8 m
   for two cars) is **wider than a traffic lane**, so normal oncoming traffic read as
   head-on collisions -> size the envelope on **lateral half-width** instead.

The final funnel on the 60 s clip:

```
390 pairs within 25 m
  -> 138 pass the TTC + lateral-envelope test
  -> 51 graded conflicts  (18 critical, 33 serious)
  -> 87 reclassified as close interactions (density signal, not safety events)
```

**Events.** `events.csv` carries a uniform schema across all detectors:
`event_type, track_id, secondary_track_id, timestamp, duration, x, y, severity,
value, unit, description`.

---

## Results (60 s, 1798 frames, urban intersection)

| | |
|---|---|
| Road users | **292** - 231 car, 18 LGV, 3 HGV, 2 bus, 9 motorcycle, 29 pedestrian |
| Motor vehicles / vulnerable users | 263 / 38 |
| Speed | mean 16.7, median 16.8, **p85 23.6** km/h |
| Queues | 91 detections, largest 22 vehicles, longest 78.7 m |
| Turning movements | 86 straight, 14 left, 14 right |
| Potential conflicts | **51** (18 critical, 33 serious) |
| Congestion | peak score 0.686 |
| Events | 387 across 8 types |

---

## Outputs

| File | Contents |
|---|---|
| `outputs/annotated.mp4` | Boxes, IDs, class, live speed, trails, conflict lines, HUD |
| `outputs/trajectories.csv` | **The core artifact** - every observation of every road user |
| `outputs/events.csv` | All detected events, uniform schema |
| `outputs/summary.json` | Headline metrics, calibration evidence, method notes |
| `outputs/track_summary.csv` | One row per road user |
| `outputs/trajectory_map.png` | Top-down trajectory plot with a metric scale bar |
| `queues / congestion / turning_movements / directional_flow / interactions / headways .csv` | Per-analytic tables |
| `outputs/boxes.csv` | Raw boxes, so video can be re-rendered without re-detecting |

---

## What we deliberately do not claim

Being explicit here is a design principle, not a disclaimer. Fabricated precision is
worse than an honest estimate.

| Area | We claim | We do **not** claim |
|---|---|---|
| **Classes** | The six COCO classes the detector actually outputs | That the model distinguishes LGV from HGV. **It does not.** That split is a footprint-size heuristic from calibration, tagged `class_source=size_heuristic_from_calibration` |
| **Speed** | "Estimated speed", from a telemetry-derived ground plane | Survey-grade. Assumes flat ground; uses the bbox bottom as the contact point. Headline figure is p85, not the jitter-sensitive raw max |
| **Stationary** | "Stationary vehicle **candidate**", cross-referenced against queues | An incident. One clip cannot separate a breakdown from a red light |
| **Wrong way** | `against_dominant_flow` - movement against the *observed* flow of that part of the road, learned from data | A legal violation. We have no HD map of legal carriageway directions |
| **Conflicts** | "**Potential** conflict", graded, from a constant-velocity CPA projection | A collision, or a certified traffic-conflict-technique score. A constant-velocity model cannot know a driver was already braking |
| **Congestion** | A transparent weighted score for operator triage | A certified level of service - that needs lane geometry and capacity |
| **Coordinates** | Metres relative to the aircraft, in one common ground frame | Absolute survey coordinates. GPS is recorded, so results *could* be georeferenced |

Additional known limitations: the gimbal sits at -63 deg (oblique, not nadir), so
projection error grows toward the far field; the flat-ground assumption ignores road
camber and gradient; and detection recall drops for small or heavily occluded objects,
which no amount of downstream analytics can recover.

---

## Project layout

```
config/config.yaml        every tunable - nothing analytical is hard-coded
config/bytetrack.yaml     tracker params (track_buffer: 90 is the key one)
pipeline/calibration.py   DJI telemetry -> metric ground projection
pipeline/detector.py      YOLOv8 loading, class mapping, colours
pipeline/tracker.py       ByteTrack, yields per-frame records
pipeline/trajectory.py    kinematics, per-track class resolution
pipeline/analytics.py     counts, queues, congestion, turning, dwell, ID stability
pipeline/interactions.py  CPA / TTC, conflict grading, headways
pipeline/anomalies.py     event detectors -> events.csv
pipeline/visualize.py     annotated video + trajectory map
run_pipeline.py           orchestration (6 stages)
app.py                     Streamlit dashboard
make_colab.py             regenerates the Colab notebook
ISSUES.md                 every problem hit, diagnosed and fixed
```

**Extending for Level 2+:** write a function that takes the trajectory DataFrame and
returns event rows, register it in `anomalies.DETECTORS`, then run
`--from-trajectories`. No detection, no tracking, no re-processing.
