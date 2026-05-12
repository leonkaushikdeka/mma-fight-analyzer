import numpy as np
from collections import deque
from enum import Enum


class FighterState(Enum):
    STANDING = "standing"
    KNOCKED_DOWN = "knocked_down"
    GETTING_UP = "getting_up"
    STANDING_UP = "standing_up"
    STALLING = "stalling"


class RefereeAI:
    def __init__(self, fps=30):
        self.fps = fps
        self.hip_ys = deque(maxlen=int(fps * 2))
        self.nose_ys = deque(maxlen=int(fps * 2))
        self.shoulder_ys = deque(maxlen=int(fps * 2))
        self.standup_tracker = deque(maxlen=int(fps * 3))

        self.state = FighterState.STANDING
        self.knockdowns = 0
        self.state_frames = 0
        self.stall_frames = 0
        self.warnings = deque(maxlen=20)
        self.last_commentary = ""
        self.commentary_printed = set()
        self._body_scale = 1.0

    def _compute_body_scale(self, landmarks):
        if landmarks is None or len(landmarks) < 25:
            return
        l_s = landmarks[11] if 11 < len(landmarks) else None
        r_s = landmarks[12] if 12 < len(landmarks) else None
        if l_s is not None and r_s is not None:
            self._body_scale = max(np.linalg.norm(l_s[:2] - r_s[:2]), 1.0)

    def _norm(self, value):
        return value / max(self._body_scale, 1.0)

    def analyze(self, landmarks, strikes_list):
        result = {
            "knockdown": False,
            "standup": False,
            "warnings": list(self.warnings)[-5:],
            "commentary": "",
            "standing": self.state == FighterState.STANDING,
            "state": self.state.value,
        }

        if landmarks is None:
            return result

        self._compute_body_scale(landmarks)
        self.state_frames += 1

        nose = self._point(landmarks, 0)
        l_hip = self._point(landmarks, 23)
        r_hip = self._point(landmarks, 24)
        l_shoulder = self._point(landmarks, 11)
        r_shoulder = self._point(landmarks, 12)

        hip_y = np.mean([p[1] for p in [l_hip, r_hip] if p is not None])
        nose_y = nose[1] if nose is not None else None
        sh_y = np.mean([p[1] for p in [l_shoulder, r_shoulder] if p is not None])

        if nose_y is not None:
            self.nose_ys.append(nose_y)
        self.hip_ys.append(hip_y)
        if sh_y is not None:
            self.shoulder_ys.append(sh_y)

        result.update(self._state_machine(strikes_list or []))
        return result

    def _state_machine(self, strikes_list):
        out = {"knockdown": False, "standup": False, "commentary": ""}

        if self.state == FighterState.STANDING:
            if self._detect_knockdown():
                self.knockdowns += 1
                self.state = FighterState.KNOCKED_DOWN
                self.state_frames = 0
                out["knockdown"] = True
                out["commentary"] = "KNOCKDOWN! Fighter is down!"
                self._warn("Knockdown detected")

        elif self.state == FighterState.KNOCKED_DOWN:
            if self._detect_standup_start():
                self.state = FighterState.GETTING_UP
                self.state_frames = 0
                out["commentary"] = "Fighter is trying to get up!"
            elif self.state_frames > self.fps * 20:
                self.state = FighterState.STANDING
                self.state_frames = 0
                out["commentary"] = "Fighter survived but that was a bad one."
                self._warn("Fighter down for 20s - stood up")

        elif self.state == FighterState.GETTING_UP:
            if self._detect_standing():
                self.state = FighterState.STANDING
                self.state_frames = 0
                out["standup"] = True
                out["commentary"] = "Fighter is back on their feet!"
            elif self.state_frames > self.fps * 10:
                self.state = FighterState.STANDING
                self.state_frames = 0
                out["standup"] = True
                out["commentary"] = "Back to the feet."

        if self.state == FighterState.STANDING:
            if self._detect_stalling(strikes_list):
                if self.state != FighterState.STALLING:
                    self._warn("Stalling warning")
                    out["commentary"] = "Warning: Stalling! Work!"
                self.state = FighterState.STALLING
                self.state_frames = 0
            else:
                if self.state == FighterState.STALLING:
                    pass
                self.state = FighterState.STANDING
        else:
            self.stall_frames = 0

        return out

    def _detect_knockdown(self):
        if len(self.nose_ys) < 15:
            return False
        recent = np.median(list(self.nose_ys)[-5:])
        sh_y = np.median(list(self.shoulder_ys)[-5:]) if len(self.shoulder_ys) > 5 else None
        if sh_y is None:
            return False
        nose_below_shoulder = (recent - sh_y) > self._body_scale * 0.3
        recent_change = abs(recent - self.nose_ys[-1]) > self._body_scale * 0.1

        if len(self.nose_ys) >= 15:
            baseline = np.median(list(self.nose_ys)[:10])
            drop_ratio = (recent - baseline) / (baseline + 1)
            return drop_ratio > 0.25 and nose_below_shoulder
        return False

    def _detect_standup_start(self):
        if len(self.hip_ys) < 10:
            return False
        recent = np.median(list(self.hip_ys)[-5:])
        low_point = min(self.hip_ys)
        rise = low_point - recent
        return self._norm(rise) > 0.15

    def _detect_standing(self):
        if len(self.shoulder_ys) < 5 or len(self.nose_ys) < 5:
            return False
        nose_mid = np.median(list(self.nose_ys)[-5:])
        sh_mid = np.median(list(self.shoulder_ys)[-5:])
        return (sh_mid - nose_mid) < self._body_scale * 0.15

    def _detect_stalling(self, strikes_list):
        if not strikes_list:
            self.stall_frames += 1
        else:
            self.stall_frames = 0
        return self.stall_frames > self.fps * 5

    def _warn(self, msg):
        if msg not in self.commentary_printed:
            self.warnings.append(msg)
            self.commentary_printed.add(msg)
            if len(self.commentary_printed) > 50:
                self.commentary_printed.clear()

    def reset_round(self):
        self.__init__(fps=self.fps)

    @staticmethod
    def _point(landmarks, idx):
        if idx < len(landmarks):
            return landmarks[idx][:2]
        return None
