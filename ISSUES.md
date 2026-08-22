# Issues found and how we solved them

Every problem hit during the build, the diagnosis, and the fix. This is the
engineering record - each entry is something we found by interrogating our own
output rather than trusting it.

---

## 1. The source video was unusable as-is

**Problem.** 4K, 6.5 GB, ~400 s. At `imgsz=1280` on CPU a single frame took ~0.9 s,
and decoding 4K alone was a bottleneck. A full run would have taken hours.

**Diagnosis.** Resolution was buying us nothing: at ~70 m altitude a car is already
~70 px long at 1920 wide, well inside YOLO's usable range.

**Fix.** Cut a 60 s clip and downscale 4K -> 1920 with ffmpeg.

```bash
ffmpeg -ss 120 -i source.MP4 -t 60 -vf scale=1920:-2 -c:v libx264 -preset ultrafast -crf 26 -an -sn data/intersection.mp4
```

**Why this is safe, not a shortcut.** Focal length *in pixels* scales with frame
width: `f_px = (focal_35mm / 36) * image_width`. Halve the width and both `f_px`
and the pixel coordinates halve, so the ground projection is **mathematically
identical**. Downscaling costs detection recall on small objects, not metric accuracy.

---

## 2. No GPU, and we could not install one

**Problem.** `torch.cuda.is_available()` returned `False`. CPU inference ran at
**1.13 fps** (0.89 s/frame).

**Attempted fix.** Install a CUDA build:
`pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126`

**It failed.**
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```
`shutil.disk_usage` showed only **6.5 GB free of 264 GB** on C:. Worse, pip had
already partially upgraded torch to `2.13.0+cpu` (`cuda compiled: None`).

**Decision: stay on CPU.** Rolling the dice on a torch reinstall with no disk
headroom risked breaking a *working* install mid-hackathon. Two things made this
an acceptable call rather than a compromise:

1. **Architecture absorbs it.** Detection is the only expensive stage and runs
   **once**. Every analytic reads `trajectories.csv`, so re-analysis is seconds
   (see issue 8). Levels 2-5 do not re-pay the detection cost.
2. **Colab covers the GPU need at zero risk.** `Traffic_Intelligence_Colab.ipynb`
   runs the identical code on a free T4 at ~20-25 fps.

**Side lesson.** A first attempt to measure disk usage with `du -sh` on the Temp
directory hung past a 120 s timeout. `shutil.disk_usage` answered instantly - use
the syscall, not a directory walk.

---

## 3. COCO has no LGV or HGV class

**Problem.** The deliverable asks for LGV vs HGV. COCO gives `truck` and `bus`,
full stop. Claiming otherwise would be fabricating a model capability.

**Fix.** Split `truck` detections by **measured ground footprint**, which the
telemetry calibration gives us for free: project both bottom corners of the box to
the ground plane and take the p90 of the resulting width across the track's life.
Above `lgv_hgv_split_m: 7.0` -> HGV, below -> LGV.

**The honesty mechanism.** Every such track carries
`class_source=size_heuristic_from_calibration` in `track_summary.csv`, and the
dashboard prints the caveat next to the class chart. It is never presented as a
model output. Result on the 60 s clip: 18 LGV, 3 HGV.

---

## 4. Per-frame class flicker double-counted road users

**Problem.** YOLO changes its mind between frames - the same vehicle is `car` in
one frame and `truck` in the next. Counting per-frame detections inflates totals
and produces nonsense class breakdowns.

**Fix.** Class is resolved **per track, once**, by majority vote across every
observation of that track (`resolve_track_classes`), then written back to all rows.
One physical road user = one track = one class = one count.

---

## 5. Conflict detection fired 12 times in 2 seconds, with an impossible TTC

**Problem.** The first smoke test reported 12 "potential conflicts" in a 2-second
window, with `min_ttc_s: 0.01` - physically impossible.

**Diagnosis.** Constant-velocity CPA amplifies speed noise. Cars travelling in a
line have a large relative-velocity magnitude `|dv|` purely from jitter, while
their actual **closing rate along the line of centres** is near zero. The CPA
formula happily returns a tiny `t_cpa` for that.

**Fix, in two parts:**

1. **Require genuine closing.** Project the relative velocity onto the line of
   centres and require it to exceed `min_closing_speed_kph: 6.0`. Plus a
   `ttc_min_s: 0.2` floor below which a value is noise by definition.
   ```python
   closing = -np.dot(dp, dv) / dist
   if rel_speed > 1e-3 and closing >= min_closing:
       t_cpa = -np.dot(dp, dv) / rel_speed**2
   ```
2. **Require persistence.** A real conflict develops over ~1 s; jitter fires for
   one frame. Count trigger frames per pair and drop anything under
   `min_persistence_frames: 5`.

**Result.** 12 -> 10 conflicts, `min_ttc` 0.01 -> 0.23 s, `frames_triggered` 6-50.

**Then we swept the threshold to check the survivors were not artifacts.** Conflict
count stayed stable from `ttc_max` 2.5 s all the way down to 1.5 s - meaning the
survivors are genuinely tight encounters, not pairs sitting just under a threshold.
Locked in `ttc_max_s: 2.5`, `min_approach_speed_kph: 10.0`.

**The sweep cost zero GPU time** - it re-ran `find_interactions` against the
existing `trajectories.csv`. That is the architecture paying off in practice.

---

## 6. On the full clip, 357 of 390 pairs were flagged as conflicts (91%)

**Problem.** The filters from issue 5 held up on 2 seconds but not on 60. Claiming
357 near-collisions in one minute is not credible.

**Diagnosis.** We looked at the distribution instead of guessing. 73% of conflicts
had TTC < 1.0 s - **not** the signature of threshold noise, which would pile up
*near* the threshold. So these were geometrically real. But:

| | median |
|---|---|
| closing speed of "critical" conflicts | **9.6 km/h** |
| miss distance at CPA | **3.34 m** |

Two vehicles converging at 2.7 m/s that will pass 3.3 m apart are **creeping in a
queue**, not in conflict. The ~1.5 s TTC threshold from the traffic-conflict
literature assumes lane-disciplined traffic; in dense mixed traffic low TTC is
*normal operating behaviour*.

**Fix: grade jointly on TTC and closing speed.** Closing speed is the proxy for how
much kinetic energy an evasive manoeuvre must absorb.

```
critical : TTC <= 1.5s AND closing >= 20 km/h
serious  : TTC <= 1.5s AND closing >= 12 km/h   (>= 8 km/h if a vulnerable user)
close    : passed every geometric filter but low-energy -> routine density
```

Only critical + serious are reported as conflicts. "Close interaction" is still
counted and exported, because in dense traffic it is a genuine *density* signal -
it just is not a safety event. We also report **rates** (per minute, per 100 road
users) so the number is comparable across clips.

**Result.** 357 -> 129 conflicts.

---

## 7. The remaining conflicts were normal oncoming traffic

**Problem.** After issue 6, one pattern stood out in the critical list: track `855`
appeared in four separate "head-on" conflicts and `663` in three. A vehicle having
four head-on conflicts in a minute is not plausible.

**First hypothesis - wrong-way driving - was wrong.** We checked the bearings:
tracks 855 and 663 average ~290 deg, and the scene's dominant flows are 270 deg and
90 deg (a two-way east-west road). 290 deg sits *inside* the main 270 deg stream.
They were not driving against traffic; they were just fast (25 and 32 km/h) and
therefore met a lot of oncoming vehicles.

**Real diagnosis.** We compared miss distance by interaction geometry:

| geometry | median `d_cpa` |
|---|---|
| crossing | 1.88 m |
| following | 1.97 m |
| merging | 2.04 m |
| **head-on** | **3.39 m** |

33 of 51 head-on conflicts had `d_cpa > 2.5 m`. A traffic lane is ~3.0-3.5 m wide
and a car is ~1.8 m wide, so **3.4 m is exactly the centre-to-centre separation of
two vehicles passing normally in adjacent opposing lanes.**

**Root cause.** The conflict envelope was built from `CLASS_RADIUS_M`, an
*omnidirectional* radius of 1.9 m for a car - effectively a half-*length*. Summed
for two cars that is a 3.8 m envelope, **wider than a lane**. Normal oncoming
traffic fell inside it.

**Fix.** Size the conflict envelope on **lateral half-width** (`CLASS_HALF_WIDTH_M`,
0.9 m for a car) with a 1.15 noise margin -> ~2.1 m for a car pair. That admits
genuine collision courses and rejects adjacent-lane passing. The omnidirectional
radius is retained for the separate sustained-proximity test, where it is correct.

**Result.** 129 -> **51 conflicts (18 critical + 33 serious)**, and every surviving
`d_cpa` is now <= 2.21 m across all four geometries. The final funnel:

```
390 pairs within 25 m
  -> 138 pass TTC + lateral-envelope test
  -> 51 graded conflicts (18 critical, 33 serious)
  -> 87 reclassified as close interactions (density, not safety)
```

The survivors are convincing: a rear-end risk at 0.86 m miss sustained 68 frames,
a bus merging at 1.0 m, a car/motorcycle crossing at 0.83 m.

---

## 8. Re-running analytics required a 25-minute re-detection

**Problem.** Fixing issues 6 and 7 changed only the analytics layer, but
`run_pipeline.py` always started from YOLO. Re-deriving the numbers meant 25 min of
CPU detection for a change that touched no pixels.

**Fix.** Added `--from-trajectories`, which skips detection and tracking entirely
and re-derives every analytic from a saved `trajectories.csv`. Bounding boxes are
now persisted to `boxes.csv` too, so even the annotated video can be re-rendered
without re-detecting.

```bash
python run_pipeline.py --from-trajectories outputs/trajectories.csv --tag graded
```

Runtime: **seconds instead of 25 minutes.** This is the trajectory-first
architecture made executable - and it is how Levels 2-5 will add analytics without
rebuilding detection.

---

## 9. A 67.7 km/h top speed at an intersection averaging 17 km/h

**Problem.** `max_speed` reported 67.7 km/h while p85 was 23.6 km/h. Either the
calibration was broken or a track was corrupted.

**Diagnosis.** Neither - it is a tail artifact, and the numbers proved it:

| percentile | speed |
|---|---|
| p50 | 3.1 km/h (includes stopped vehicles) |
| p85 | 20.4 km/h |
| p99 | 32.8 km/h |
| p99.9 | 37.2 km/h |
| max | 67.7 km/h |

Only **13 observations out of 69,067 (0.019%)**, across 4 of 292 tracks, exceeded
40 km/h - a few frames of box jitter on fast-moving vehicles.

**Fix.** Report `p85_speed` as the headline figure (which is also what traffic
engineering actually uses) and add `robust_max_speed_p99_5`. The raw max is retained
in `summary.json` with an explicit
`max_speed_caveat: "raw max is jitter-sensitive; use p85 / p99.5 for reporting"`.
We did not silently clip the data - we changed which statistic we quote.

---

## 10. Track IDs did not survive occlusion

**Problem.** Default ByteTrack settings dropped IDs when a vehicle passed behind a
tree or bus, or sat still long enough at a red light to be treated as gone. Each
recovery became a *new* ID, inflating counts and fragmenting trajectories.

**Fix.** The single highest-value tuning change was `track_buffer: 90` in
`config/bytetrack.yaml` - ~3 s at 30 fps - plus a lowered `track_low_thresh: 0.08`
so weak detections still feed re-association.

**Measured result** on the 60 s clip: **2,688 gap recoveries** (a track re-linked
after missed detections rather than reborn), longest bridged gap **3.07 s**, mean
track life 9.85 s, longest 59.96 s - i.e. tracks surviving the entire clip.

**Why we report this metric.** We have no hand-labelled ground truth, so we cannot
compute MOTA or IDF1, and we do not quote one. Gap recoveries are the honest,
directly measurable evidence for stable identity.

---

## 11. Stationary vehicles were being reported as incidents

**Problem.** A vehicle stopped for 20 s at a red light is not a breakdown, but a
naive stationary detector reports both identically.

**Fix.** Stationary detection cross-references the queue detector. If a stopped
vehicle is inside a detected queue cluster, the event description says so
explicitly. The event type is `stationary_vehicle_candidate` - never
"stalled vehicle" or "incident". One clip cannot distinguish the two causes, so we
surface the evidence and let a human decide.

---

## 12. Wrong-way detection with no HD map

**Problem.** Detecting illegal wrong-way driving requires knowing the legal
direction of each carriageway. We have no HD map.

**Fix.** Learn the dominant flow **from the data**: bin the scene into a grid and
compute the circular mean bearing of all traffic per cell. Flag tracks deviating
more than `wrong_way_deviation_deg: 130` from their own cell's dominant flow.

**The honest framing.** The event type is `against_dominant_flow`, not
"wrong_way_violation". It is a flow anomaly. This also correctly avoids false
positives on legitimate turning traffic, which follows the local flow.

---

## 13. `half=True` spammed a deprecation warning every frame

**Problem.** Newer Ultralytics warns `'half' is deprecated ... Use 'quantize'
instead` on every `model.track()` call - 1,798 lines of noise, hiding real output.

**Fix.** Only pass the key when it is actually enabled, so CPU runs never trigger it:
```python
if det_settings.get("half"):
    kw["half"] = True
```

---

## 14. RuntimeWarnings from short tracks

**Problem.** `Mean of empty slice` / `All-NaN axis encountered` from
`np.nanmean`/`np.nanmax` on tracks too short to have valid smoothed kinematics.

**Assessment.** Cosmetic - the NaN propagates correctly and those tracks are
filtered by `min_track_frames: 8` downstream. Suppressed at the run level with
`python -W ignore` rather than spending build time on it. Logged here because it
is a real (if harmless) rough edge, not something we pretend is clean.

---

## 15. A background run looked hung, and wasn't

**Problem.** The full run's log file sat at 0 bytes for minutes. It looked stalled.

**Diagnosis.** Not a hang - `grep -vE` in the pipeline was **buffering** stdout.
`tasklist | grep python` showed a live process at 245,800 K RSS, working normally.

**Lesson.** Verify with process state, not log output, before killing a long job.

---

## 16. The annotated video would not play in a browser

**Problem.** OpenCV writes `mp4v`, which browsers and the Streamlit player refuse.
The 1920p output was also 382 MB.

**Fix.** Re-encode to H.264 once, and have the dashboard prefer `annotated_web.mp4`
when it exists, falling back to the raw file and showing the exact ffmpeg command
if it does not.

```bash
ffmpeg -i outputs/annotated.mp4 -vcodec libx264 -crf 28 -pix_fmt yuv420p outputs/annotated_web.mp4
```

---

## 17. Seeding SAM with trajectory points segmented the cars, not the road

**Problem.** Road segmentation was added so analytics could use the *carriageway* as a
denominator instead of the whole frame (a frame is mostly rooftops and vegetation).
MobileSAM was chosen: ~40 MB, already inside `ultralytics`, no new dependency, and
because the drone **hovers** it only has to run on a single frame.

SAM needs prompts. The obvious choice was to prompt it at the trajectory points -
after all, those are on the road. The mask came back with a measured agreement of
**IoU 0.342**, and SAM covered only **35.8%** of the area traffic had demonstrably
driven over, while its own masks covered just 10.5% of the frame.

**Diagnosis.** The tell was in the log line: *24 prompts -> 24 segments*. Exactly one
segment per prompt. SAM is **class-agnostic** - it returns *the object under the
point*, and a trajectory point sits on a **vehicle**. We had asked SAM to segment the
road and it had faithfully segmented 24 cars.

**Fix.** Put the prompts on **bare carriageway**:

- exclude any candidate covered by a detection box in the frame being segmented -
  `boxes.csv` already records exactly that, and because the aircraft hovers, road a
  vehicle drove over at t=30 s is bare tarmac at t=0;
- take at most one candidate per 96 px grid cell, so prompts spread across every arm
  of the junction instead of clustering in the busiest lane.

SAM's coverage of the travelled area went **35.8% -> 56.7%**, and its own mask area
went 10.5% -> 27.6% - now comparable to the 26.1% the traffic actually used.

---

## 18. The road mask then included a rooftop and a tree canopy

**Problem.** With better seeds, SAM contributed much more area - and the union of
"SAM" plus "where traffic drove" now contained a large grey **rooftop** and a big
**tree canopy**. Visible immediately in the overlay image.

**Diagnosis.** Same root cause as #17, pointing the other way. SAM has no concept of
"road". Prompted on tarmac, it grows the segment by appearance, and a grey rooftop
looks like grey tarmac from 70 m up. This is not a tuning problem - a class-agnostic
model cannot be tuned into having a class.

The consequence would have been silent and bad: the whole point of the mask is to
report density **per square metre of road**, and a rooftop in the denominator makes
that number meaningless.

**Fix.** Stop treating SAM as an authority and make it a **proposer**:

> **SAM proposes, the observed traffic vouches.**

Keep the segments separately rather than merging them, then accept a segment only if
at least 45% of *its own area* lies inside the travelled mask. A road segment
extending an in-use corridor into its empty lanes passes. A rooftop overlaps the
travelled area at roughly zero and is rejected.

On this footage **13 of 28 segments were rejected**, IoU rose **0.342 -> 0.468**, and
the final mask covers 27.2% of the frame against the 26.1% traffic proved - i.e.
deliberately conservative. The asymmetry is intentional: an over-inclusive road mask
is worse than a slightly tight one.

Off-road detections are **dimmed in the video but kept in the analytics** by default,
because a detection on a verge or footpath may well be a real road user - and
pedestrians are exempt from filtering entirely. Deleting vulnerable road users to
tidy up a mask would be exactly the wrong trade for a safety system.

---

## 19. "Small object detection" is a claim, so we measured it

**Problem.** Aerial video is fundamentally a small-object problem: at 70 m a
motorcycle is a few dozen pixels. The textbook responses are to infer at higher
resolution and to lower the confidence floor. Both also buy false positives, so
"we enabled small-object detection" is not a result.

**Diagnosis.** `tools/measure_small_objects.py` runs three settings over the same 30
frames sampled evenly across the clip, and reports detections bucketed by **pixel
area** plus the fraction landing **inside the road mask** - reusing the segmentation
from #17/#18, so the two features validate each other.

| Setting | Detections | tiny (<400 px) | small | medium | large | on-road | median conf |
|---|---|---|---|---|---|---|---|
| 1280, conf 0.25 (baseline) | 1,250 | 92 | 503 | 636 | 19 | **87.6%** | 0.522 |
| **1920, conf 0.25** | **1,975** (+58%) | **535** | 743 | 677 | 20 | **79.5%** | 0.469 |
| 1920, conf 0.15 | 3,164 (+153%) | 1,140 | 1,055 | 916 | 53 | **64.9%** | 0.315 |

**What decided it.** Not the headline count - the *shape* of the change.

Going 1280 -> 1920 added **+443 tiny** and +240 small detections but only **+41
medium and +1 large**. That is precisely what extra resolution should do if the gain
is real: more pixels cannot reveal a bus you were already seeing. Tiny detections
rose **5.8x** (92 -> 535), and 79.5% of all detections still landed on the
carriageway.

Dropping conf to 0.15 behaves like noise instead. It added +280 medium and +34
**large** detections - resolution was unchanged, so a lower floor cannot be finding
genuinely new large vehicles, it is admitting duplicates and junk. On-road fraction
collapses to **64.9%**: more than a third of detections sit on rooftops and
vegetation, which is the signature of hallucination, not recall.

**Fix.** Adopt `imgsz: 1920` at `conf: 0.25`. **Reject `conf: 0.15`** despite it
having by far the biggest detection count.

The cost is honest: 1920 inference is roughly 2.2x slower per frame than 1280, which
on a CPU-only machine is the difference between a ~25 minute and a ~55 minute pass.
On the Colab T4 it is free. `summary.json` records `imgsz` for every run, so which
setting produced which artifact is never ambiguous.

We do **not** claim a recall or precision figure. There is no hand-labelled ground
truth for this clip, so what is honestly available is the measured delta plus the
evidence about its plausibility - which is what the table above is.

---

## 20. Trajectory trails vanished behind the vehicle

**Problem.** Trails were drawn over a fixed 45-frame window, so a car's path faded
out about 1.5 s behind it. Watching the video, you could not see that an ID had
survived a long occlusion - which is the single thing Level 1 is scored on.

**Diagnosis.** Not a bug; the window was chosen to stop the frame turning into
spaghetti with 30+ simultaneous tracks. Both requirements are real and they conflict.

**Fix.** Two layers instead of one. The **full history from first detection** is drawn
thin at 45% brightness, so continuity is visible end to end; the **recent 45 frames**
are drawn bright and thickening on top, so current motion still reads clearly. Cheap
- both come from the same `trail_by_track` array already in memory.

---

## The pattern behind all of these

Nine of the twenty issues above (4, 5, 6, 7, 9, 11, 17, 18, 19) were found by
**interrogating our own output** - looking at distributions, cross-tabulating by
category, and asking whether a number was physically plausible - not by seeing a
crash.

The conflict detector is the clearest example. It never threw an error. It produced
confident, well-formatted, plausible-looking output at every stage, and it was
wrong three times in a row for three completely different reasons: speed noise
(issue 5), then dense-traffic norms (issue 6), then a geometry error that made
normal oncoming traffic look like head-on collisions (issue 7). Each one was only
findable by asking "is this number actually believable?" and then checking a
distribution instead of trusting the answer.

The segmentation work repeated the pattern in miniature, twice in a row and in
opposite directions - SAM segmenting cars instead of road (17), then swallowing a
rooftop (18) - and both were caught by a single measured number, the overlap between
what SAM proposed and where traffic demonstrably drove.

And issue 19 is the same discipline applied *before* believing good news: the setting
with the largest improvement in detection count is the one we rejected.
