# MMA Fight Analyzer

Real-time pose-based MMA analysis, strike detection, referee AI, and coaching suggestions — **100% offline**.

Detects punches, kicks, knockdowns, guard quality, forward pressure, and generates referee-style commentary and coaching tips from a webcam or video file using MediaPipe pose estimation.

## Features

| Module | What it does |
|--------|-------------|
| **Pose Estimation** | 33-point body landmark tracking via MediaPipe |
| **Movement Analysis** | Stance detection, forward pressure, head movement, footwork score, guard quality |
| **Strike Detection** | Jab, cross, hooks, uppercuts, front kicks, roundhouse kicks, knees — speed & landing detection |
| **Referee AI** | Knockdown detection, standup detection, stalling warnings, illegal strike warnings |
| **Scoring Engine** | MMA judging criteria: effective striking, aggression, round scoring |
| **Suggestion Engine** | Real-time coaching: guard position, head movement, striking variety, pressure management |
| **Visualizer** | OpenCV HUD with metrics panel, color-coded status, round timer |

## Requirements

- Python 3.9+
- Webcam or video file
- No internet required during runtime

## Install

```bash
# Clone
git clone https://github.com/leonkaushikdeka/mma-fight-analyzer.git
cd mma-fight-analyzer

# Install
pip install -r requirements.txt
```

## Usage

```bash
# Webcam (default)
python main.py

# Video file
python main.py --source path/to/fight.mp4

# Custom resolution
python main.py --width 1920 --height 1080

# Help
python main.py --help
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause/Resume |
| `r` | Reset round stats |

## How It Works

```
Webcam/Video Frame
       │
       ▼
MediaPipe Pose (33 landmarks) ◄── runs offline on CPU/GPU
       │
       ├──► Movement Analyzer     ── stance, speed, pressure, guard
       ├──► Strike Detector       ── hand/foot velocity + trajectory
       ├──► Referee AI            ── hip height drop → knockdown, stalling
       ├──► Scoring Engine        ── MMA judging criteria
       └──► Suggestion Engine     ── pattern-based coaching
       │
       ▼
OpenCV HUD (metrics panel + pose overlay)
```

### Strike Detection Logic

- Tracks wrist/ankle velocity over consecutive frames
- Classifies trajectory (forward / lateral / upward)
- Maps to strike type based on which hand/leg and direction
- Calculates speed in px/frame and estimates landing proximity

### Referee AI Logic

- Monitors nose height baseline → sudden drop = knockdown
- Tracks hip height after knockdown → rise = standup
- Counts idle frames → stalling warning
- Checks wrist position vs belt line → low blow warning

## Project Structure

```
mma-fight-analyzer/
├── main.py                 # Entry point
├── requirements.txt
├── README.md
├── src/
│   ├── pose_estimation.py  # MediaPipe wrapper
│   ├── movement_analyzer.py
│   ├── strike_detector.py
│   ├── scoring_engine.py
│   ├── referee_ai.py
│   ├── suggestion_engine.py
│   └── visualizer.py
└── data/
    └── sample/
```

## Roadmap

- [x] Single-fighter pose tracking
- [ ] Two-fighter tracking (sparring partner)
- [ ] Takedown / clinch detection
- [ ] Round timer + automatic round transitions
- [ ] CSV/JSON match log export
- [ ] Pre-recorded fight video analysis
- [ ] Lightweight ML model for strike classification
