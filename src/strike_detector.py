import numpy as np
from collections import deque
import warnings


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


class LimbBuffer:
    def __init__(self, window=5):
        self.positions = deque(maxlen=window)
        self.timestamps = deque(maxlen=window)

    def push(self, pos, timestamp):
        self.positions.append(np.array(pos, dtype=np.float32))
        self.timestamps.append(timestamp)

    @property
    def count(self):
        return len(self.positions)

    @property
    def latest(self):
        return self.positions[-1] if self.positions else None

    def smoothed_velocity(self, delta_frames=3):
        if self.count < delta_frames + 1:
            return 0.0, np.array([0.0, 0.0], dtype=np.float32)
        p_curr = self.positions[-1][:2]
        p_prev = self.positions[-(delta_frames + 1)][:2]
        dt = (self.timestamps[-1] - self.timestamps[-(delta_frames + 1)])
        if dt <= 0:
            return 0.0, np.array([0.0, 0.0], dtype=np.float32)
        delta = p_curr - p_prev
        speed = np.linalg.norm(delta) / dt
        return speed, delta

    def peak_velocity(self, delta_frames=3):
        if self.count < delta_frames + 2:
            return 0.0
        velocities = []
        for i in range(delta_frames, self.count):
            p1 = self.positions[i - delta_frames][:2]
            p2 = self.positions[i][:2]
            dt = self.timestamps[i] - self.timestamps[i - delta_frames]
            if dt > 0:
                velocities.append(np.linalg.norm(p2 - p1) / dt)
        return max(velocities) if velocities else 0.0


class StrikeDetector:
    def __init__(self, fps=30):
        self.fps = fps
        self.window = max(3, int(fps * 0.1))
        self.vel_delta = max(2, int(fps * 0.067))

        self.l_buffers = {
            "wrist": LimbBuffer(self.window),
            "ankle": LimbBuffer(self.window),
            "elbow": LimbBuffer(self.window),
            "knee": LimbBuffer(self.window),
            "shoulder": LimbBuffer(self.window),
            "hip": LimbBuffer(self.window),
        }
        self.r_buffers = {
            "wrist": LimbBuffer(self.window),
            "ankle": LimbBuffer(self.window),
            "elbow": LimbBuffer(self.window),
            "knee": LimbBuffer(self.window),
            "shoulder": LimbBuffer(self.window),
            "hip": LimbBuffer(self.window),
        }

        self.ts = 0.0
        self.strikes_thrown = 0
        self.strikes_landed = 0
        self.strike_log = deque(maxlen=100)
        self.strike_counts = {k: 0 for k in STRIKE_TYPES}
        self.strike_speeds = deque(maxlen=60)
        self.recent_strikes = deque(maxlen=10)

        self.left_is_lead = True
        self.lead_confirmed = False

        self._cooldowns = {"hand": 0, "leg": 0}
        self._last_hand_positions = {}
        self._body_scale = 1.0

    def _body_scale_from_landmarks(self, landmarks):
        if landmarks is None or len(landmarks) < 25:
            return 1.0
        l_s = landmarks[11] if 11 < len(landmarks) else None
        r_s = landmarks[12] if 12 < len(landmarks) else None
        l_h = landmarks[23] if 23 < len(landmarks) else None
        r_h = landmarks[24] if 24 < len(landmarks) else None
        d = 0.0
        count = 0
        if l_s is not None and r_s is not None:
            d += np.linalg.norm(l_s[:2] - r_s[:2])
            count += 1
        if l_h is not None and r_h is not None:
            d += np.linalg.norm(l_h[:2] - r_h[:2])
            count += 1
        scale = (d / max(count, 1)) if count > 0 else 1.0
        self._body_scale = max(scale, 1.0)
        return self._body_scale

    def _normalize(self, value):
        return value / max(self._body_scale, 1.0)

    def _extract(self, landmarks):
        idx_map = {
            "l_wrist": 15, "r_wrist": 16,
            "l_elbow": 13, "r_elbow": 14,
            "l_shoulder": 11, "r_shoulder": 12,
            "l_hip": 23, "r_hip": 24,
            "l_knee": 25, "r_knee": 26,
            "l_ankle": 27, "r_ankle": 28,
            "nose": 0, "l_ear": 7, "r_ear": 8,
        }
        pts = {}
        for name, idx in idx_map.items():
            if idx < len(landmarks):
                pts[name] = landmarks[idx][:2]
        return pts

    def analyze(self, landmarks):
        if landmarks is None:
            self.ts += 1.0 / self.fps
            for side in ("l", "r"):
                for buf in self.l_buffers.values() if side == "l" else self.r_buffers.values():
                    buf.push([0, 0], self.ts)
            for k in self._cooldowns:
                self._cooldowns[k] = max(0, self._cooldowns[k] - 1)
            return self._empty_result()

        self._body_scale_from_landmarks(landmarks)
        pts = self._extract(landmarks)
        self.ts += 1.0 / self.fps

        self._update_buffers(pts)
        self._resolve_stance(pts)

        for k in self._cooldowns:
            self._cooldowns[k] = max(0, self._cooldowns[k] - 1)

        strikes = []
        strikes.extend(self._detect_hand_strikes(pts))
        strikes.extend(self._detect_leg_strikes(pts))

        for s in strikes:
            self.strikes_thrown += 1
            if s["landed"]:
                self.strikes_landed += 1
            if s["type"] in self.strike_counts:
                self.strike_counts[s["type"]] += 1
            self.strike_speeds.append(s["speed"])
            self.recent_strikes.append(s)

        return {
            "strikes": strikes,
            "total_thrown": self.strikes_thrown,
            "total_landed": self.strikes_landed,
            "accuracy": round(self.strikes_landed / max(self.strikes_thrown, 1) * 100, 1),
            "strike_counts": dict(self.strike_counts),
            "avg_speed": round(np.mean(self.strike_speeds) if self.strike_speeds else 0, 1),
            "output_rate": round(len([s for s in self.strike_log if hasattr(s, 'get') and s.get("ts", 0) > self.ts - 60]) / 60.0, 1) if self.strike_log else 0,
            "body_scale": round(self._body_scale, 1),
        }

    def _update_buffers(self, pts):
        mapping = {
            "wrist": ("l_wrist", "r_wrist"),
            "ankle": ("l_ankle", "r_ankle"),
            "elbow": ("l_elbow", "r_elbow"),
            "knee": ("l_knee", "r_knee"),
            "shoulder": ("l_shoulder", "r_shoulder"),
            "hip": ("l_hip", "r_hip"),
        }
        for buf_name, (l_key, r_key) in mapping.items():
            l_val = pts.get(l_key)
            r_val = pts.get(r_key)
            if l_val is not None:
                self.l_buffers[buf_name].push(l_val, self.ts)
            if r_val is not None:
                self.r_buffers[buf_name].push(r_val, self.ts)

    def _resolve_stance(self, pts):
        l_w = pts.get("l_wrist")
        r_w = pts.get("r_wrist")
        l_s = pts.get("l_shoulder")
        r_s = pts.get("r_shoulder")
        if l_w is None or r_w is None or l_s is None or r_s is None:
            return
        if not self.lead_confirmed:
            l_dist = np.linalg.norm(l_w - l_s) if l_s is not None else 0
            r_dist = np.linalg.norm(r_w - r_s) if r_s is not None else 0
            if abs(l_dist - r_dist) > 20 and l_dist > 0 and r_dist > 0:
                self.left_is_lead = l_dist < r_dist
                self.lead_confirmed = True

    def _detect_hand_strikes(self, pts):
        if self._cooldowns["hand"] > 0:
            return []
        strikes = []

        for side_prefix, bufs in [("l", self.l_buffers), ("r", self.r_buffers)]:
            wrist = bufs["wrist"]
            elbow = bufs["elbow"]
            shoulder = bufs["shoulder"]
            if wrist.count < self.vel_delta + 1:
                continue

            speed, delta = wrist.smoothed_velocity(self.vel_delta)
            norm_speed = self._normalize(speed)
            threshold = 3.0

            if norm_speed < threshold:
                continue

            wrist_pos = pts.get(f"{side_prefix}_wrist")
            sh_pos = pts.get(f"{side_prefix}_shoulder")
            el_pos = pts.get(f"{side_prefix}_elbow")
            nose = pts.get("nose")

            if wrist_pos is None or sh_pos is None:
                continue

            is_lead_hand = (side_prefix == "l") == self.left_is_lead
            punch_type, confidence = self._classify_punch(
                delta, sh_pos, wrist_pos, el_pos, is_lead_hand, pts
            )

            if punch_type and confidence > 0.4:
                landed = self._check_landing(wrist_pos, nose, sh_pos)
                strikes.append({
                    "type": punch_type,
                    "side": "lead" if is_lead_hand else "rear",
                    "speed": round(speed, 1),
                    "norm_speed": round(norm_speed, 2),
                    "landed": landed,
                    "power": STRIKE_TYPES.get(punch_type, {}).get("power", 0.5),
                    "ts": self.ts,
                })
                self._cooldowns["hand"] = int(self.fps * 0.12)

        return strikes

    def _classify_punch(self, delta, shoulder, wrist, elbow, is_lead, pts):
        if shoulder is None or wrist is None:
            return None, 0.0
        shoulder_to_wrist = np.array(wrist) - np.array(shoulder)
        sw_norm = np.linalg.norm(shoulder_to_wrist)
        if sw_norm < 1:
            return None, 0.0
        direction = shoulder_to_wrist / sw_norm
        alignment = np.dot(delta[:2], direction) / (np.linalg.norm(delta[:2]) + 1e-8)

        lateral = abs(delta[0]) > abs(delta[1]) * 1.2
        upward = delta[1] < -abs(delta[0]) * 0.4
        forward = alignment > 0.3

        if not forward and not lateral and not upward:
            return None, 0.0

        if elbow is not None:
            ext = self._angle_at(wrist, elbow, shoulder)
        else:
            ext = 90

        if forward and alignment > 0.6 and ext > 120:
            return ("jab" if is_lead else "cross"), alignment
        elif lateral and ext > 60:
            return ("lead_hook" if is_lead else "rear_hook"), min(alignment + 0.3, 1.0)
        elif upward and ext < 100 and not lateral:
            return ("lead_uppercut" if is_lead else "rear_uppercut"), min(abs(alignment) + 0.2, 1.0)
        elif forward and alignment > 0.3 and ext > 90:
            return ("jab" if is_lead else "cross"), alignment * 0.7

        return None, 0.0

    def _detect_leg_strikes(self, pts):
        if self._cooldowns["leg"] > 0:
            return []
        strikes = []

        for side_prefix, bufs in [("l", self.l_buffers), ("r", self.r_buffers)]:
            ankle = bufs["ankle"]
            knee = bufs["knee"]
            hip = bufs["hip"]
            if ankle.count < self.vel_delta + 1:
                continue

            speed, delta = ankle.smoothed_velocity(self.vel_delta)
            norm_speed = self._normalize(speed)
            if norm_speed < 2.5:
                continue

            knee_pos = pts.get(f"{side_prefix}_knee")
            hip_pos = pts.get(f"{side_prefix}_hip")
            ankle_pos = pts.get(f"{side_prefix}_ankle")
            if knee_pos is None or hip_pos is None or ankle_pos is None:
                continue

            knee_angle = self._angle_at(ankle_pos, knee_pos, hip_pos)
            hip_to_ankle = np.array(ankle_pos) - np.array(hip_pos)
            ha_norm = np.linalg.norm(hip_to_ankle)
            if ha_norm < 1:
                continue
            direction = hip_to_ankle / ha_norm
            alignment = np.dot(delta[:2], direction) / (np.linalg.norm(delta[:2]) + 1e-8)

            lateral = abs(delta[0]) > abs(delta[1]) * 1.2
            forward = alignment > 0.3
            upward = delta[1] < -abs(delta[0]) * 0.4

            strike_type = None
            if forward and alignment > 0.5 and knee_angle > 120:
                strike_type = "front_kick"
            elif lateral and norm_speed > 3.5:
                strike_type = "roundhouse"
            elif upward and knee_angle < 90:
                strike_type = "knee"

            if strike_type:
                strikes.append({
                    "type": strike_type,
                    "side": side_prefix,
                    "speed": round(speed, 1),
                    "norm_speed": round(norm_speed, 2),
                    "landed": False,
                    "power": STRIKE_TYPES.get(strike_type, {}).get("power", 0.5),
                    "ts": self.ts,
                })
                self._cooldowns["leg"] = int(self.fps * 0.25)

        return strikes

    def _check_landing(self, wrist, nose, shoulder):
        if wrist is None or nose is None:
            return False
        dist = np.linalg.norm(np.array(wrist) - np.array(nose))
        return dist < self._body_scale * 0.6

    def _empty_result(self):
        return {
            "strikes": [],
            "total_thrown": self.strikes_thrown,
            "total_landed": self.strikes_landed,
            "accuracy": round(self.strikes_landed / max(self.strikes_thrown, 1) * 100, 1),
            "strike_counts": dict(self.strike_counts),
            "avg_speed": round(np.mean(self.strike_speeds) if self.strike_speeds else 0, 1),
            "output_rate": 0,
            "body_scale": round(self._body_scale, 1),
        }

    @staticmethod
    def _angle_at(p1, p2, p3):
        a = np.array(p1[:2], dtype=np.float32)
        b = np.array(p2[:2], dtype=np.float32)
        c = np.array(p3[:2], dtype=np.float32)
        ba = a - b
        bc = c - b
        dot = np.dot(ba, bc)
        norm = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
        return np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))
