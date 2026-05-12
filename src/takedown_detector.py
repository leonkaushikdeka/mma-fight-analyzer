import numpy as np
from collections import deque
from enum import Enum


class FightPhase(Enum):
    STANDING = "standing"
    SHOOTING = "shooting"
    CLINCH = "clinch"
    TAKEDOWN = "takedown"
    GROUND = "ground"


class TakedownDetector:
    def __init__(self, fps=30):
        self.fps = fps
        self.hip_y_buffer = deque(maxlen=int(fps * 1.5))
        self.hip_x_buffer = deque(maxlen=int(fps * 0.5))
        self.shoulder_y_buffer = deque(maxlen=int(fps * 0.5))
        self.phase = FightPhase.STANDING
        self.phase_frames = 0
        self.takedowns = 0
        self.takedown_attempts = 0
        self.clinch_entries = 0
        self.ground_control_time = 0.0
        self._last_hip_x = None
        self._body_scale = 1.0
        self._cooldown = 0

    def analyze(self, landmarks_a, landmarks_b=None, body_scale=1.0):
        self._body_scale = max(body_scale, 1.0)
        self.phase_frames += 1
        self._cooldown = max(0, self._cooldown - 1)

        if landmarks_a is None:
            return self._result(0, 0)

        hip_center = self._midpoint(landmarks_a, 23, 24)
        shoulder_center = self._midpoint(landmarks_a, 11, 12)
        if hip_center is None:
            return self._result(0, 0)

        hip_y = hip_center[1]
        hip_x = hip_center[0]
        sh_y = shoulder_center[1] if shoulder_center else None

        self.hip_y_buffer.append(hip_y)
        if len(self.hip_y_buffer) > 5:
            self.hip_x_buffer.append(hip_x)
        if sh_y is not None:
            self.shoulder_y_buffer.append(sh_y)

        fighter_b_close = self._check_proximity(landmarks_a, landmarks_b)
        self._update_phase(hip_center, shoulder_center, fighter_b_close)

        return self._result(hip_center, shoulder_center)

    def _update_phase(self, hip_center, shoulder_center, fighter_b_close):
        if self._cooldown > 0:
            return

        if len(self.hip_y_buffer) < 10:
            return

        hip_arr = np.array(list(self.hip_y_buffer))
        baseline = np.median(hip_arr[:5])
        recent = np.median(hip_arr[-5:])
        drop = (recent - baseline) / self._body_scale

        if self.phase in (FightPhase.STANDING, FightPhase.CLINCH):
            if drop > 0.25 and fighter_b_close:
                self.phase = FightPhase.SHOOTING
                self.phase_frames = 0
                self.takedown_attempts += 1
            elif fighter_b_close and drop > 0.1:
                self.phase = FightPhase.CLINCH
                self.clinch_entries += 1
            elif drop > 0.4:
                self.phase = FightPhase.SHOOTING
                self.phase_frames = 0
                self.takedown_attempts += 1

        elif self.phase == FightPhase.SHOOTING:
            if drop > 0.45 and self.phase_frames > int(self.fps * 0.2):
                self.phase = FightPhase.TAKEDOWN
                self.takedowns += 1
                self._cooldown = int(self.fps * 2)
            elif abs(drop) < 0.05 and self.phase_frames > int(self.fps * 1):
                self.phase = FightPhase.STANDING

        elif self.phase == FightPhase.TAKEDOWN:
            if hip_arr[-1] > baseline - self._body_scale * 0.1:
                self.phase = FightPhase.GROUND
                self.phase_frames = 0
            elif self.phase_frames > int(self.fps * 3):
                self.phase = FightPhase.GROUND

        elif self.phase == FightPhase.GROUND:
            self.ground_control_time += 1.0 / self.fps
            if hip_arr[-1] < baseline - self._body_scale * 1.0:
                self.phase = FightPhase.STANDING

    def _check_proximity(self, la, lb):
        if la is None or lb is None:
            return False
        hip_a = self._midpoint(la, 23, 24)
        hip_b = self._midpoint(lb, 23, 24)
        if hip_a is None or hip_b is None:
            return False
        dist = np.linalg.norm(np.array(hip_a) - np.array(hip_b))
        return dist < self._body_scale * 3.0

    def _result(self, hip_center, shoulder_center):
        torso_angle = 0.0
        if hip_center is not None and shoulder_center is not None:
            dx = shoulder_center[0] - hip_center[0]
            dy = shoulder_center[1] - hip_center[1]
            if abs(dy) > 1:
                torso_angle = float(np.degrees(np.arctan2(dx, dy)))

        return {
            "phase": self.phase.value,
            "takedowns": self.takedowns,
            "takedown_attempts": self.takedown_attempts,
            "clinch_entries": self.clinch_entries,
            "ground_control_time": round(self.ground_control_time, 1),
            "torso_angle": round(torso_angle, 1),
        }

    def reset_round(self):
        self.__init__(fps=self.fps)

    @staticmethod
    def _midpoint(landmarks, idx_a, idx_b):
        if landmarks is None:
            return None
        a = landmarks[idx_a] if idx_a < len(landmarks) else None
        b = landmarks[idx_b] if idx_b < len(landmarks) else None
        if a is None or b is None:
            return None
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
