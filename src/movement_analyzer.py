import numpy as np
from collections import deque


JOINTS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_foot": 31, "right_foot": 32,
    "nose": 0,
}


class MovementAnalyzer:
    def __init__(self, history_frames=15, fps=30):
        self.fps = fps
        self.history_len = history_frames
        self.position_history = deque(maxlen=history_frames)
        self.stance = None
        self.forward_pressure = 0.0
        self.head_movement_score = 0.0
        self.footwork_score = 0.0
        self.total_distance = 0.0
        self.last_hip_center = None

    def analyze(self, landmarks):
        if landmarks is None:
            return {}

        h, w = landmarks.shape[:2] if landmarks.ndim > 1 else (1, 1)
        pts = {}
        for name, idx in JOINTS.items():
            if idx < len(landmarks):
                pts[name] = landmarks[idx][:2]

        self.position_history.append(pts)
        return self._compute_metrics()

    def _compute_metrics(self):
        if len(self.position_history) < 3:
            return {}

        current = self.position_history[-1]
        early = self.position_history[0]
        metrics = {}

        hip_center = self._midpoint(current, "left_hip", "right_hip")
        if hip_center is not None and self.last_hip_center is not None:
            dist = np.linalg.norm(np.array(hip_center) - np.array(self.last_hip_center))
            self.total_distance += dist
            metrics["speed"] = round(dist * self.fps, 1)
        self.last_hip_center = hip_center

        metrics["total_distance"] = round(self.total_distance, 1)

        self.stance = self._detect_stance(current)
        metrics["stance"] = self.stance

        self.forward_pressure = self._compute_forward_pressure()
        metrics["forward_pressure"] = round(self.forward_pressure, 2)

        self.head_movement_score = self._compute_head_movement()
        metrics["head_movement"] = round(self.head_movement_score, 2)

        self.footwork_score = self._compute_footwork()
        metrics["footwork"] = round(self.footwork_score, 2)

        metrics["guard_position"] = self._guard_quality(current)

        return metrics

    def _detect_stance(self, pts):
        l_s = pts.get("left_shoulder")
        r_s = pts.get("right_shoulder")
        l_w = pts.get("left_wrist")
        r_w = pts.get("right_wrist")
        if l_s is None or r_s is None:
            return "unknown"
        dx = abs(l_s[0] - r_s[0])
        if l_w and r_w:
            if l_w[1] < r_w[1] - 10:
                return "orthodox"
            elif r_w[1] < l_w[1] - 10:
                return "southpaw"
        if l_s[0] < r_s[0]:
            return "orthodox (facing right)"
        return "southpaw (facing left)"

    def _compute_forward_pressure(self):
        if len(self.position_history) < self.history_len // 2:
            return 0.0
        recent = self.position_history[-1]
        oldest = self.position_history[0]
        nose_now = recent.get("nose")
        nose_old = oldest.get("nose")
        if nose_now is None or nose_old is None:
            return 0.0
        return float(nose_old[1] - nose_now[1])

    def _compute_head_movement(self):
        if len(self.position_history) < 5:
            return 0.0
        nose_positions = []
        for frame in self.position_history:
            n = frame.get("nose")
            if n is not None:
                nose_positions.append(n)
        if len(nose_positions) < 5:
            return 0.0
        arr = np.array(nose_positions)
        lateral = np.std(arr[:, 0])
        vertical = np.std(arr[:, 1])
        return float(lateral + vertical * 0.5)

    def _compute_footwork(self):
        if len(self.position_history) < self.history_len:
            return 0.0
        ankle_dists = []
        for frame in self.position_history:
            l_a = frame.get("left_ankle")
            r_a = frame.get("right_ankle")
            if l_a is not None and r_a is not None:
                ankle_dists.append(np.linalg.norm(np.array(l_a) - np.array(r_a)))
        if not ankle_dists:
            return 0.0
        return float(np.std(ankle_dists))

    def _guard_quality(self, pts):
        nose = pts.get("nose")
        l_w = pts.get("left_wrist")
        r_w = pts.get("right_wrist")
        if nose is None or l_w is None or r_w is None:
            return "unknown"
        avg_hand_y = (l_w[1] + r_w[1]) / 2
        if avg_hand_y < nose[1] - 20:
            return "high"
        elif avg_hand_y < nose[1] + 30:
            return "mid"
        else:
            return "low"

    @staticmethod
    def _midpoint(pts, name_a, name_b):
        a = pts.get(name_a)
        b = pts.get(name_b)
        if a is None or b is None:
            return None
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

    @staticmethod
    def angle(p1, p2, p3):
        a = np.array(p1[:2]) if len(p1) > 2 else np.array(p1)
        b = np.array(p2[:2]) if len(p2) > 2 else np.array(p2)
        c = np.array(p3[:2]) if len(p3) > 2 else np.array(p3)
        ba = a - b
        bc = c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
