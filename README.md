# MMA Fight Analyzer

Real-time pose-based MMA analysis with strike detection, referee AI, and coaching suggestions. **100% offline** — no cloud, no API keys, no internet required.

Uses [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) for 33-point body tracking and geometric heuristics for strike classification, knockdown detection, and movement analysis. Runs on CPU or GPU.

## How It Works

```
Camera frame
    │
    ▼
MediaPipe Pose (33 landmarks) ─── body scale computed per frame
    │
    ├──► Movement Analyzer   stance (orthodox/southpaw), guard height,
    │                         forward pressure, head movement, footwork
    │                         ── all normalized by shoulder width
    │
    ├──► Strike Detector     separate limb buffers per side, smoothed
    │                         velocity over N-frame window, trajectory
    │                         classification (forward/lateral/upward),
    │                         elbow angle checks, body-scale thresholds
    │
    ├──► Referee AI          state machine: STANDING → KNOCKED_DOWN →
    │                         GETTING_UP → STANDING, body-relative
    │                         height drop detection, stalling monitor
    │
    ├──► Scoring Engine      MMA criteria: effective striking, aggression
    │
    └──► Suggestion Engine   pattern-based coaching: guard, pressure,
                              strike variety, head movement, feinting
    │
    ▼
    OpenCV HUD (metrics panel + pose skeleton overlay)
```

### Body-Scale Normalization

All thresholds are computed relative to the person's shoulder width per frame. This means:

- Detection works the same at 480p, 720p, or 1080p
- Camera distance changes don't break thresholds
- A child and an adult are treated proportionally

### Strike Detection Approach

| Step | What happens |
|------|-------------|
| 1 | Each wrist/ankle has its own FIFO buffer of recent positions |
| 2 | Velocity = position(t) - position(t-N) / time, where N = 2-3 frames |
| 3 | Movement vector dot product with shoulder-to-wrist direction = alignment score |
| 4 | Forward punch with alignment > 0.6 + elbow angle > 120° = jab/cross |
| 5 | Lateral movement with speed > 3.5x body scale = hook |
| 6 | Upward movement with elbow angle < 100° = uppercut |
| 7 | Leg velocity check + knee angle for kicks/knees |

### Referee AI State Machine

```
            ┌───────────────────────────────────────┐
            │                                       │
            ▼                                       │
  ┌─────────────────┐    nose drops >25%      ┌──────────┐
  │    STANDING     │ ──────────────────────► │ KNOCKED  │
  │  (normal ops)   │                         │  DOWN    │
  └─────────────────┘ ◄────────────────────── └──────────┘
            │  back to feet     hip rises >15%      │
            │                                        │
            │  no strikes for 5s                     │
            ├────────────────────────────────────────► STALLING
            │                                        │
            ▼                                        ▼
     throws a strike                            standup
```

## Requirements

- Python 3.9+
- Webcam or video file
- No internet during runtime

## Install

```bash
git clone https://github.com/leonkaushikdeka/mma-fight-analyzer.git
cd mma-fight-analyzer
pip install -r requirements.txt
```

## Usage

```bash
# Webcam (default)
python main.py

# Video file
python main.py --source path/to/fight.mp4

# Performance mode: process every 2nd frame (skip=1)
python main.py --skip 1

# Max accuracy: full model + no frame skipping
python main.py --complexity 2

# Low-end hardware: lite model + frame skipping
python main.py --complexity 0 --skip 2

# Custom resolution
python main.py --width 1920 --height 1080
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause / Resume |
| `r` | Reset round stats |
| `+` / `=` | Increase frame skipping (more performance) |
| `-` / `_` | Decrease frame skipping (more accuracy) |

## Architecture

```
mma-fight-analyzer/
├── main.py                  Entry point
├── requirements.txt         mediapipe, opencv-python, numpy
├── README.md
├── src/
│   ├── main.py              CLI, main loop, error recovery
│   ├── pose_estimation.py   MediaPipe wrapper, 33-point landmark extraction
│   ├── movement_analyzer.py Stance, guard, pressure, head movement (normalized)
│   ├── strike_detector.py   Per-limb buffered velocity, trajectory classification
│   ├── scoring_engine.py    MMA round scoring
│   ├── referee_ai.py        State machine: knockdown/standup/stalling detection
│   ├── suggestion_engine.py Pattern-based real-time coaching
│   └── visualizer.py        OpenCV HUD with metrics panel
└── data/
    └── sample/
```

## Key Design Decisions

- **Body scale normalization** via shoulder width per frame — thresholds are resolution-independent
- **Separate limb buffers** — left/right wrists, ankles, elbows tracked independently
- **Velocity over N-frame window** — 2-3 frame delta for smoothing, not single-frame diff
- **State machine for knockdowns** — avoids false positives from bending over
- **Cooldown per limb group** — prevents double-counting the same strike
- **Frame skipping** — configurable via CLI or +/- keys at runtime

## Roadmap

- [x] Single-fighter tracking
- [x] Body-scale normalized metrics
- [x] Per-limb strike buffers
- [x] Referee state machine
- [x] Performance mode (frame skipping)
- [ ] Two-fighter sparring tracking
- [ ] Takedown/clinch detection
- [ ] Automated round transitions
- [ ] CSV match log export
- [ ] Pre-recorded fight video analysis
