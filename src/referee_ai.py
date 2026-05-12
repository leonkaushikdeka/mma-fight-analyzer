import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RefereeResult:
    knockdown_detected: bool = False
    standup_detected: bool = False
    stalling_warning: bool = False
    low_blow_warning: bool = False
    illegal_strike: Optional[str] = None
    timestamp: float = 0.0


KNOCKDOWN_THRESHOLD = 0.4


class RefereeAI:
    def __init__(self, fps=30, history_seconds=3):
        self.fps = fps
        self.history_len = int(fps * history_seconds)
        self.hip_heights = deque(maxlen=self.history_len)
        self.nose_heights = deque(maxlen=self.history_len)
        self.standup_history = deque(maxlen=self.history_len)
        self.knockdowns = 0
        self.standing_eight_counts = 0
        self.warnings = deque(maxlen=20)
        self.standing = True
        self.downed_alert = False
        self.stall_warning = False
        self.stall_frames = 0
        self.last_commentary = ""
        self.commentary_cooldown = 0

    def analyze(self, landmarks, strikes):
        import time
        result = RefereeResult(timestamp=time.time())

        if landmarks is None:
            return result

        hip_y = self._get_joint_y(landmarks, [23, 24])
        nose_y = self._get_joint_y(landmarks, [0])
        nose_x = self._get_point(landmarks, 0)

        if hip_y is not None:
            self.hip_heights.append(hip_y)
        if nose_y is not None:
            self.nose_heights.append(nose_y)

        self._check_knockdown(result, nose_y, nose_x[0] if nose_x is not None else 0)
        self._check_standup(result, hip_y)
        self._check_stalling(strikes, result)
        self._check_illegal_strikes(strikes, landmarks, result)
        self._check_illegal_strikes_type(strikes, landmarks, result)

        if self.commentary_cooldown > 0:
            self.commentary_cooldown -= 1

        return result

    def detect_knockdown(self, landmarks, baseline_nose_height):
        nose_y = self._get_joint_y(landmarks, [0])
        if nose_y is None or baseline_nose_height == 0:
            return False, 0.0
        drop = (nose_y - baseline_nose_height) / baseline_nose_height
        return drop > KNOCKDOWN_THRESHOLD, drop

    def detect_standup(self, landmarks, prev_hip_height):
        hip_y = self._get_joint_y(landmarks, [23, 24])
        if hip_y is None or prev_hip_height is None:
            return False
        return abs(hip_y - prev_hip_height) > 30

    def check_stalling(self, landmarks, history, threshold_frames):
        return self.stall_frames > threshold_frames

    def check_low_blow(self, landmarks):
        belt_line = self._get_joint_y(landmarks, [23, 24])
        l_wrist = self._get_point(landmarks, 15)
        r_wrist = self._get_point(landmarks, 16)
        if belt_line is None:
            return False
        if l_wrist is not None and l_wrist[1] > belt_line + 20:
            return True
        if r_wrist is not None and r_wrist[1] > belt_line + 20:
            return True
        return False

    def check_illegal_strike(self, landmarks):
        return None

    def _check_knockdown(self, result, nose_y, nose_x):
        if nose_y is None or len(self.nose_heights) < 10:
            return
        recent = np.mean(list(self.nose_heights)[-5:])
        baseline = np.mean(list(self.nose_heights)[:10])
        if baseline == 0:
            return
        drop = (recent - baseline) / baseline

        if drop > KNOCKDOWN_THRESHOLD and not self.downed_alert:
            self.knockdowns += 1
            result.knockdown_detected = True
            self.standing = False
            self.downed_alert = True
            result.illegal_strike = "Knockdown! Fighter is down!"
            self.warnings.append("Knockdown detected")
            self.last_commentary = result.illegal_strike

    def _check_standup(self, result, hip_y):
        if hip_y is None:
            return
        self.standup_history.append(hip_y)

        if self.downed_alert and len(self.standup_history) > 10:
            recent = np.mean(list(self.standup_history)[-5:])
            low_point = min(self.standup_history)
            if abs(recent - low_point) > 30:
                result.standup_detected = True
                self.standing = True
                self.downed_alert = False
                result.illegal_strike = "Fighter is back on their feet!"
                self.warnings.append("Standup detected")
                self.last_commentary = result.illegal_strike

    def _check_stalling(self, strikes, result):
        if strikes is None or len(strikes) == 0:
            self.stall_frames += 1
        else:
            self.stall_frames = 0
            self.stall_warning = False

        if self.stall_frames > self.fps * 5 and not self.stall_warning:
            self.stall_warning = True
            result.stalling_warning = True
            result.illegal_strike = "Warning: Stalling! Work!"
            self.warnings.append("Stalling warning")
            self.last_commentary = result.illegal_strike

    def _check_illegal_strikes(self, strikes, landmarks, result):
        if not strikes:
            return
        belt_line = self._get_joint_y(landmarks, [23, 24])
        for s in strikes:
            if belt_line is None:
                continue
            wrist = self._get_point(landmarks, 15 if s.get("side") == "left" else 16)
            if wrist is not None and wrist[1] > belt_line + 20:
                result.low_blow_warning = True
                result.illegal_strike = "Low blow warning!"

    def _check_illegal_strikes_type(self, strikes, landmarks, result):
        pass

    def reset_round(self):
        self.hip_heights.clear()
        self.nose_heights.clear()
        self.standup_history.clear()
        self.downed_alert = False
        self.standing = True
        self.stall_warning = False
        self.stall_frames = 0

    @staticmethod
    def _get_joint_y(landmarks, indices):
        valid = []
        for i in indices:
            if i < len(landmarks):
                valid.append(landmarks[i][1])
        return np.mean(valid) if valid else None

    @staticmethod
    def _get_point(landmarks, idx):
        if idx < len(landmarks):
            return landmarks[idx][:2]
        return None
