import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class StrikeResult:
    strike_type: str
    hand_foot: str
    velocity: float
    landed: bool
    timestamp: float
    confidence: float


STRIKE_TYPES = {
    "jab": {"hand": "lead", "direction": "forward", "power": 0.5},
    "cross": {"hand": "rear", "direction": "forward", "power": 0.9},
    "lead_hook": {"hand": "lead", "direction": "lateral", "power": 0.8},
    "rear_hook": {"hand": "rear", "direction": "lateral", "power": 0.9},
    "lead_uppercut": {"hand": "lead", "direction": "upward", "power": 0.7},
    "rear_uppercut": {"hand": "rear", "direction": "upward", "power": 0.85},
    "front_kick": {"hand": None, "direction": "forward", "power": 0.7},
    "roundhouse": {"hand": None, "direction": "lateral", "power": 0.85},
    "knee": {"hand": None, "direction": "upward", "power": 0.9},
}


class StrikeDetector:
    def __init__(self, history_frames=5, velocity_threshold=8.0, fps=30):
        self.fps = fps
        self.velocity_threshold = velocity_threshold
        self.history_len = history_frames
        self.wrist_history = deque(maxlen=history_frames)
        self.ankle_history = deque(maxlen=history_frames)
        self.elbow_history = deque(maxlen=history_frames)
        self.knee_history = deque(maxlen=history_frames)
        self.shoulder_history = deque(maxlen=history_frames)
        self.hip_history = deque(maxlen=history_frames)

        self.strikes_thrown = 0
        self.strikes_landed = 0
        self.strike_log = deque(maxlen=50)
        self.strike_counts = {k: 0 for k in STRIKE_TYPES}
        self.strike_speeds = deque(maxlen=30)
        self.recent_strikes = deque(maxlen=10)
        self.cooldown_counter = 0
        self.last_hand_positions = {"left": None, "right": None}

    def analyze(self, landmarks):
        if landmarks is None:
            self.cooldown_counter = max(0, self.cooldown_counter - 1)
            return {"strikes": []}

        pts = self._extract_points(landmarks)
        self._store_history(pts)
        self.cooldown_counter = max(0, self.cooldown_counter - 1)
        strikes = self._detect_strikes(pts)

        hand_positions = self._get_hand_positions(pts)
        for side, pos in hand_positions.items():
            self.last_hand_positions[side] = pos

        return {
            "strikes": strikes,
            "total_thrown": self.strikes_thrown,
            "total_landed": self.strikes_landed,
            "accuracy": round(self.strikes_landed / max(self.strikes_thrown, 1) * 100, 1),
            "strike_counts": dict(self.strike_counts),
            "avg_speed": round(np.mean(self.strike_speeds) if self.strike_speeds else 0, 1),
            "output_rate": round(len([s for s in self.strike_log if s["time_since"] < 60]) / 60.0 * self.fps, 1) if False else 0,
        }

    def detect_punch(self, wrist_pos, elbow_pos, shoulder_pos, prev_pos):
        if wrist_pos is None or prev_pos is None:
            return None, 0.0
        delta = np.array(wrist_pos[:2]) - np.array(prev_pos[:2])
        speed = np.linalg.norm(delta) * self.fps
        if speed < self.velocity_threshold:
            return None, speed
        is_forward = False
        if shoulder_pos is not None and wrist_pos is not None:
            s2w = np.array(wrist_pos[:2]) - np.array(shoulder_pos[:2])
            is_forward = np.dot(delta, s2w) > 0
        is_lateral = abs(delta[0]) > abs(delta[1]) * 1.5
        is_upward = delta[1] < -abs(delta[0]) * 0.5
        if is_forward and speed > self.velocity_threshold * 1.2:
            return "jab", speed
        elif is_lateral and speed > self.velocity_threshold * 1.5:
            return "hook", speed
        elif is_upward and speed > self.velocity_threshold:
            return "uppercut", speed
        return None, speed

    def detect_kick(self, ankle_pos, hip_pos, prev_pos):
        if ankle_pos is None or prev_pos is None:
            return None, 0.0
        delta = np.array(ankle_pos[:2]) - np.array(prev_pos[:2])
        speed = np.linalg.norm(delta) * self.fps
        if speed < self.velocity_threshold * 0.8:
            return None, speed
        is_forward = False
        if hip_pos is not None:
            h2a = np.array(ankle_pos[:2]) - np.array(hip_pos[:2])
            is_forward = np.dot(delta, h2a) > 0
        is_lateral = abs(delta[0]) > abs(delta[1]) * 1.5
        if is_forward and speed > self.velocity_threshold:
            return "front_kick", speed
        elif is_lateral and speed > self.velocity_threshold * 1.2:
            return "roundhouse", speed
        return None, speed

    def detect_knee(self, knee_pos, hip_pos):
        if knee_pos is None or hip_pos is None:
            return None, 0.0
        return None, 0.0

    def compute_velocity(self, current, previous, fps):
        if current is None or previous is None:
            return 0.0
        delta = np.array(current[:2]) - np.array(previous[:2])
        return float(np.linalg.norm(delta) * fps)

    def detect_landing(self, strike_pos, target_pos, threshold=80.0):
        if strike_pos is None or target_pos is None:
            return False
        dist = np.linalg.norm(strike_pos[:2] - target_pos[:2])
        return dist < threshold

    def _extract_points(self, landmarks):
        idx_map = {
            "l_wrist": 15, "r_wrist": 16,
            "l_elbow": 13, "r_elbow": 14,
            "l_shoulder": 11, "r_shoulder": 12,
            "l_hip": 23, "r_hip": 24,
            "l_knee": 25, "r_knee": 26,
            "l_ankle": 27, "r_ankle": 28,
            "l_foot": 31, "r_foot": 32,
            "nose": 0,
            "l_index": 19, "r_index": 20,
        }
        pts = {}
        for name, idx in idx_map.items():
            if idx < len(landmarks):
                pts[name] = landmarks[idx]
        return pts

    def _store_history(self, pts):
        for key, container in [
            ("l_wrist", self.wrist_history), ("r_wrist", self.wrist_history),
            ("l_ankle", self.ankle_history), ("r_ankle", self.ankle_history),
            ("l_elbow", self.elbow_history), ("r_elbow", self.elbow_history),
            ("l_knee", self.knee_history), ("r_knee", self.knee_history),
            ("l_shoulder", self.shoulder_history), ("r_shoulder", self.shoulder_history),
            ("l_hip", self.hip_history), ("r_hip", self.hip_history),
        ]:
            if key in pts:
                container.append(pts[key])

    def _get_hand_positions(self, pts):
        return {
            "left": pts.get("l_wrist"),
            "right": pts.get("r_wrist"),
        }

    def _velocity(self, history):
        if len(history) < 2:
            return 0.0, np.array([0.0, 0.0])
        p1 = history[-2][:2]
        p2 = history[-1][:2]
        delta = np.array(p2) - np.array(p1)
        speed = np.linalg.norm(delta) * self.fps
        return speed, delta

    def _is_forward(self, delta, shoulder_pos, wrist_pos):
        if shoulder_pos is None or wrist_pos is None:
            return True
        shoulder_to_wrist = np.array(wrist_pos[:2]) - np.array(shoulder_pos[:2])
        return np.dot(delta, shoulder_to_wrist) > 0

    def _is_lateral(self, delta):
        return abs(delta[0]) > abs(delta[1]) * 1.5

    def _is_upward(self, delta):
        return delta[1] < -abs(delta[0]) * 0.5

    def _detect_strikes(self, pts):
        if self.cooldown_counter > 0:
            return []
        strikes = []

        l_wrist = pts.get("l_wrist")
        r_wrist = pts.get("r_wrist")
        l_ankle = pts.get("l_ankle")
        r_ankle = pts.get("r_ankle")
        l_knee = pts.get("l_knee")
        r_knee = pts.get("r_knee")

        hands = [
            ("left", l_wrist, pts.get("l_elbow"), pts.get("l_shoulder")),
            ("right", r_wrist, pts.get("r_elbow"), pts.get("r_shoulder")),
        ]
        for side, wrist, elbow, shoulder in hands:
            if wrist is None:
                continue
            speed, delta = self._velocity(self.wrist_history)
            if speed < self.velocity_threshold:
                continue
            strike_type = None
            if self._is_forward(delta, shoulder, wrist) and speed > self.velocity_threshold * 1.2:
                strike_type = "jab" if side == "left" else "cross"
            elif self._is_lateral(delta) and speed > self.velocity_threshold * 1.5:
                strike_type = f"{'lead' if side == 'left' else 'rear'}_hook"
            elif self._is_upward(delta) and speed > self.velocity_threshold:
                strike_type = f"{'lead' if side == 'left' else 'rear'}_uppercut"

            if strike_type:
                landed = self._check_landing(wrist, pts.get("nose"), pts)
                strike_info = {
                    "type": strike_type,
                    "side": side,
                    "speed": round(speed, 1),
                    "landed": landed,
                    "power": STRIKE_TYPES.get(strike_type, {}).get("power", 0.5),
                }
                strikes.append(strike_info)
                self.strikes_thrown += 1
                if landed:
                    self.strikes_landed += 1
                if strike_type in self.strike_counts:
                    self.strike_counts[strike_type] += 1
                self.strike_speeds.append(speed)
                self.recent_strikes.append(strike_info)
                self.cooldown_counter = int(self.fps * 0.15)

        for side, ankle, knee, hip in [
            ("left", l_ankle, l_knee, pts.get("l_hip")),
            ("right", r_ankle, r_knee, pts.get("r_hip")),
        ]:
            if ankle is None or knee is None:
                continue
            speed, delta = self._velocity(self.ankle_history)
            if speed < self.velocity_threshold * 0.8:
                continue
            strike_type = None
            if self._is_forward(delta, hip, knee) and speed > self.velocity_threshold:
                strike_type = "front_kick"
            elif self._is_lateral(delta) and speed > self.velocity_threshold * 1.2:
                strike_type = "roundhouse"
            elif self._is_upward(delta) and self._angle_at(knee, hip, ankle) < 90:
                strike_type = "knee"

            if strike_type:
                strike_info = {
                    "type": strike_type,
                    "side": side,
                    "speed": round(speed, 1),
                    "landed": False,
                    "power": STRIKE_TYPES.get(strike_type, {}).get("power", 0.5),
                }
                strikes.append(strike_info)
                self.strikes_thrown += 1
                if strike_type in self.strike_counts:
                    self.strike_counts[strike_type] += 1
                self.strike_speeds.append(speed)
                self.recent_strikes.append(strike_info)
                self.cooldown_counter = int(self.fps * 0.2)

        return strikes

    def _check_landing(self, wrist, nose, pts):
        if wrist is None or nose is None:
            return False
        dist = np.linalg.norm(wrist[:2] - nose[:2])
        return dist < 80

    def _angle_at(self, joint, a, b):
        v1 = np.array(joint[:2]) - np.array(a[:2]) if a is not None else np.array([0, 0])
        v2 = np.array(joint[:2]) - np.array(b[:2]) if b is not None else np.array([0, 1])
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        return np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))
