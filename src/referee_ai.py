import numpy as np
from collections import deque


KNOCKDOWN_THRESHOLD = 0.4
ILLEGAL_TARGETS = {
    "back_of_head": False,
    "groin": False,
    "spine": False,
    "kidneys": False,
}


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
        if landmarks is None:
            return {
                "knockdown": False,
                "standup": False,
                "warnings": [],
                "commentary": "",
                "standing": self.standing,
            }

        results = {
            "knockdown": False,
            "standup": False,
            "warnings": list(self.warnings),
            "commentary": "",
            "standing": self.standing,
        }

        hip_y = self._get_joint_y(landmarks, [23, 24])
        nose_y = self._get_joint_y(landmarks, [0])
        nose_x = self._get_point(landmarks, 0)

        if hip_y is not None:
            self.hip_heights.append(hip_y)
        if nose_y is not None:
            self.nose_heights.append(nose_y)

        self._check_knockdown(results, nose_y, nose_x if nose_x is not None else 0)
        self._check_standup(results, hip_y)
        self._check_stalling(strikes, results)
        self._check_illegal_strikes(strikes, landmarks, results)

        if self.commentary_cooldown > 0:
            self.commentary_cooldown -= 1

        return results

    def _check_knockdown(self, results, nose_y, nose_x):
        if nose_y is None or len(self.nose_heights) < 10:
            return
        recent = np.mean(list(self.nose_heights)[-5:])
        baseline = np.mean(list(self.nose_heights)[:10])
        if baseline == 0:
            return
        drop = (recent - baseline) / baseline

        if drop > KNOCKDOWN_THRESHOLD and not self.downed_alert:
            self.knockdowns += 1
            results["knockdown"] = True
            self.standing = False
            self.downed_alert = True
            results["commentary"] = "KNOCKDOWN! Fighter is down!"
            self.warnings.append("Knockdown detected")
            self.last_commentary = results["commentary"]

    def _check_standup(self, results, hip_y):
        if hip_y is None:
            return
        self.standup_history.append(hip_y)

        if self.downed_alert and len(self.standup_history) > 10:
            recent = np.mean(list(self.standup_history)[-5:])
            low_point = min(self.standup_history)
            if abs(recent - low_point) > 30:
                results["standup"] = True
                self.standing = True
                self.downed_alert = False
                results["commentary"] = "Fighter is back on their feet!"
                self.warnings.append("Standup detected")
                self.last_commentary = results["commentary"]

    def _check_stalling(self, strikes, results):
        if strikes is None or len(strikes) == 0:
            self.stall_frames += 1
        else:
            self.stall_frames = 0
            self.stall_warning = False

        if self.stall_frames > self.fps * 5 and not self.stall_warning:
            self.stall_warning = True
            results["commentary"] = "Warning: Stalling! Work!"
            self.warnings.append("Stalling warning")
            self.last_commentary = results["commentary"]

    def _check_illegal_strikes(self, strikes, landmarks, results):
        if not strikes:
            return
        belt_line = self._get_joint_y(landmarks, [23, 24])
        for s in strikes:
            if belt_line is None:
                continue
            wrist = self._get_point(landmarks, 15 if s.get("side") == "left" else 16)
            if wrist is not None and wrist[1] > belt_line + 20:
                self.warnings.append("Potential low blow detected!")
                results["commentary"] = "Watch the belt line!"

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
