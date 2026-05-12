import time
from collections import deque


ROUND_DURATION = 300
REST_DURATION = 60
NUM_ROUNDS = 5
WARNING_TIME = 10


class RoundManager:
    def __init__(self, num_rounds=NUM_ROUNDS, round_duration=ROUND_DURATION, rest_duration=REST_DURATION):
        self.num_rounds = num_rounds
        self.round_duration = round_duration
        self.rest_duration = rest_duration
        self.current_round = 1
        self.phase = "pre_fight"
        self.phase_start = time.time()
        self.round_start = 0.0
        self.round_end = 0.0
        self.round_history = deque(maxlen=20)
        self.bell_played = False
        self.horn_played = False

    def update(self, now=None):
        if now is None:
            now = time.time()
        elapsed = now - self.phase_start
        result = {
            "round": self.current_round,
            "phase": self.phase,
            "phase_time": elapsed,
            "round_time_remaining": 0,
            "bell": False,
            "horn": False,
            "match_over": False,
        }

        if self.phase == "pre_fight":
            if elapsed >= 3:
                self._start_round(now)

        elif self.phase == "fighting":
            remaining = self.round_duration - elapsed
            result["round_time_remaining"] = max(0, remaining)
            if remaining <= 0:
                self._end_round(now)
                result["horn"] = True
                self.horn_played = True
            elif remaining <= WARNING_TIME and not self.bell_played:
                result["bell"] = True
                self.bell_played = True

        elif self.phase == "rest":
            remaining = self.rest_duration - elapsed
            if remaining <= 0:
                if self.current_round < self.num_rounds:
                    self.current_round += 1
                    self._start_round(now)
                    result["bell"] = True
                else:
                    self.phase = "match_over"
                    result["match_over"] = True
                    result["horn"] = True

        elif self.phase == "match_over":
            result["match_over"] = True

        return result

    def _start_round(self, now):
        self.phase = "fighting"
        self.phase_start = now
        self.round_start = now
        self.bell_played = False
        self.horn_played = False

    def _end_round(self, now):
        self.round_end = now
        self.round_history.append({
            "round": self.current_round,
            "duration": self.round_duration,
        })
        if self.current_round < self.num_rounds:
            self.phase = "rest"
            self.phase_start = now
        else:
            self.phase = "match_over"

    def phase_time_remaining(self, now=None):
        if now is None:
            now = time.time()
        elapsed = now - self.phase_start
        if self.phase == "fighting":
            return max(0, self.round_duration - elapsed)
        elif self.phase == "rest":
            return max(0, self.rest_duration - elapsed)
        return 0

    def format_time(self, seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}:{s:02d}"
