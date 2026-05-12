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
    "nose": 0, "left_ear": 7, "right_ear": 8,
}


class MovementAnalyzer:
    def __init__(self, history_frames=15, fps=30):
        self.fps = fps
        self.history_len = history_frames
        self.position_history = deque(maxlen=history_frames)
        self.stance = "unknown"
        self.forward_pressure = 0.0
        self.head_movement_score = 0.0
        self.footwork_score = 0.0
        self.total_distance = 0.0
        self.last_hip_center = None
        self._body_scale = 1.0

    def _compute_scale(self, landmarks):
        if landmarks is None or len(landmarks) < 25:
            return
        l_s = landmarks[11] if 11 < len(landmarks) else None
        r_s = landmarks[12] if 12 < len(landmarks) else None
        l_h = landmarks[23] if 23 < len(landmarks) else None
        r_h = landmarks[24] if 24 < len(landmarks) else None
        d = 0.0
        c = 0
        if l_s is not None and r_s is not None:
            d += np.linalg.norm(l_s[:2] - r_s[:2])
            c += 1
        if l_h is not None and r_h is not None:
            d += np.linalg.norm(l_h[:2] - r_h[:2])
            c += 1
        self._body_scale = max(d / max(c, 1), 1.0)

    def _norm(self, value):
        return value / max(self._body_scale, 1.0)

    def analyze(self, landmarks):
        if landmarks is None:
            return {}

        self._compute_scale(landmarks)
        h, w = landmarks.shape[:2] if landmarks.ndim > 1 else (1, 1)
        pts = {}
        for name, idx in JOINTS.items():
            if idx < len(landmarks):
                pts[name] = landmarks[idx][:2]

        self.position_history.append(pts)
        return self._compute_metrics(pts)

    def _compute_metrics(self, pts):
        if len(self.position_history) < 3:
            return {}

        metrics = {}

        hip_center = self._midpoint(pts, "left_hip", "right_hip")
        if hip_center is not None and self.last_hip_center is not None:
            dist = np.linalg.norm(np.array(hip_center) - np.array(self.last_hip_center))
            self.total_distance += dist
            metrics["speed"] = round(self._norm(dist) * self.fps, 2)
        self.last_hip_center = hip_center

        metrics["total_distance"] = round(self._norm(self.total_distance), 1)
        metrics["body_scale"] = round(self._body_scale, 1)

        self.stance = self._detect_stance(pts)
        metrics["stance"] = self.stance

        self.forward_pressure = self._compute_forward_pressure()
        metrics["forward_pressure"] = round(self._norm(self.forward_pressure), 3)

        self.head_movement_score = self._compute_head_movement()
        metrics["head_movement"] = round(self._norm(self.head_movement_score), 3)

        self.footwork_score = self._compute_footwork()
        metrics["footwork"] = round(self._norm(self.footwork_score), 3)

        metrics["guard_position"] = self._guard_quality(pts)

        return metrics

    def _detect_stance(self, pts):
        l_w = pts.get("left_wrist")
        r_w = pts.get("right_wrist")
        l_s = pts.get("left_shoulder")
        r_s = pts.get("right_shoulder")
        l_e = pts.get("left_elbow")
        r_e = pts.get("right_elbow")
        if l_s is None or r_s is None:
            return "unknown (no shoulders)"
        if l_w is not None and r_w is not None and l_e is not None and r_e is not None:
            l_reach = np.linalg.norm(np.array(l_w) - np.array(l_s))
            r_reach = np.linalg.norm(np.array(r_w) - np.array(r_s))
            diff = abs(l_reach - r_reach)
            if diff > self._body_scale * 0.15:
                if l_reach < r_reach:
                    return "orthodox"
                else:
                    return "southpaw"
        if l_w is not None and r_w is not None:
            if l_w[1] < r_w[1] - self._body_scale * 0.05:
                return "orthodox (lead hand high)"
            elif r_w[1] < l_w[1] - self._body_scale * 0.05:
                return "southpaw (lead hand high)"
        return "unknown (facing square)"

    def _compute_forward_pressure(self):
        if len(self.position_history) < self.history_len // 2:
            return 0.0
        recent_hips = []
        old_hips = []
        mid = len(self.position_history) // 2
        for i, frame in enumerate(self.position_history):
            hc = self._midpoint(frame, "left_hip", "right_hip")
            if hc is None:
                continue
            if i < mid:
                old_hips.append(hc)
            else:
                recent_hips.append(hc)
        if not recent_hips or not old_hips:
            return 0.0
        recent_center = np.mean(recent_hips, axis=0)
        old_center = np.mean(old_hips, axis=0)
        return float(old_center[1] - recent_center[1])

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
        if len(self.position_history) < 5:
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
        diff = nose[1] - avg_hand_y
        ratio = diff / max(self._body_scale, 1.0)
        if ratio > 0.15:
            return "high"
        elif ratio > -0.1:
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
