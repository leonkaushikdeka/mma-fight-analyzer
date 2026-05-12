import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class CoachingSuggestion:
    category: str
    message: str
    priority: str
    timestamp: float


GUARD_SUGGESTIONS = {
    "low": "Keep your hands up! You're leaving your chin exposed.",
    "mid": "Tuck your chin, keep those hands active.",
    "high": "Good guard - sharp eyes on the opponent.",
}

STANCE_SUGGESTIONS = {
    "unknown": "Find your stance - shoulders sideways, hands up.",
    "orthodox": "Left foot forward, right hand loaded for the cross.",
    "southpaw": "Right foot forward, left hand loaded.",
}

HEAD_MOVEMENT_LOW = 0.5
HEAD_MOVEMENT_HIGH = 3.0
FORWARD_PRESSURE_LOW = 1.0
FOOTWORK_LOW = 1.0


class SuggestionEngine:
    def __init__(self):
        self.last_suggestions = []
        self.suggestion_cooldown = 0
        self._last_shown = {}

    def analyze(self, movement, strikes, referee) -> list:
        suggestions = []
        now = time.time()

        guard = getattr(movement, "guard_quality", "unknown") if not isinstance(movement, dict) else movement.get("guard_position", "unknown")
        msg = self.check_guard_drop(guard)
        if msg and not self._cooldown_active("guard", now):
            suggestions.append(CoachingSuggestion("guard", msg, "high", now))
            self._last_shown["guard"] = now

        head_movement = getattr(movement, "head_movement_score", 0) if not isinstance(movement, dict) else movement.get("head_movement", 0)
        msg = self.check_head_movement(head_movement)
        if msg and not self._cooldown_active("movement", now):
            suggestions.append(CoachingSuggestion("movement", msg, "medium", now))
            self._last_shown["movement"] = now

        if strikes:
            msg = self.check_striking_variety(strikes)
            if msg and not self._cooldown_active("variety", now):
                suggestions.append(CoachingSuggestion("variety", msg, "medium", now))
                self._last_shown["variety"] = now

        msg = self.check_pressure(movement)
        if msg and not self._cooldown_active("pressure", now):
            suggestions.append(CoachingSuggestion("pressure", msg, "medium", now))
            self._last_shown["pressure"] = now

        if referee:
            kd = getattr(referee, "knockdown_detected", False)
            if kd and not self._cooldown_active("defense", now):
                suggestions.append(CoachingSuggestion("defense", "Protect yourself at all times! Recover and survive.", "high", now))
                self._last_shown["defense"] = now

        return suggestions

    def generate(self, movement_data, strike_data, referee_data, elapsed_seconds):
        if self.suggestion_cooldown > 0:
            self.suggestion_cooldown -= 1
            return []
        self.suggestion_cooldown = 3

        suggestions = []

        guard = movement_data.get("guard_position", "unknown") if isinstance(movement_data, dict) else getattr(movement_data, "guard_quality", "unknown")
        if guard in GUARD_SUGGESTIONS:
            suggestions.append(GUARD_SUGGESTIONS[guard])

        stance = movement_data.get("stance", "unknown") if isinstance(movement_data, dict) else getattr(movement_data, "stance", "unknown")
        if "unknown" in stance:
            suggestions.append(STANCE_SUGGESTIONS["unknown"])

        head_movement = movement_data.get("head_movement", 0) if isinstance(movement_data, dict) else getattr(movement_data, "head_movement_score", 0)
        if head_movement < HEAD_MOVEMENT_LOW:
            suggestions.append("Move your head! Slip and roll after combinations.")
        elif head_movement > HEAD_MOVEMENT_HIGH:
            suggestions.append("Great head movement - keep varying the rhythm.")

        forward_pressure = movement_data.get("forward_pressure", 0) if isinstance(movement_data, dict) else getattr(movement_data, "forward_pressure", 0)
        if forward_pressure > FORWARD_PRESSURE_LOW * 3:
            suggestions.append("Cut off the cage - don't let them circle out.")
        elif forward_pressure < FORWARD_PRESSURE_LOW:
            suggestions.append("Push forward! You're giving ground.")

        footwork = movement_data.get("footwork", 0) if isinstance(movement_data, dict) else getattr(movement_data, "footwork_score", 0)
        if footwork < FOOTWORK_LOW:
            suggestions.append("Stay light on your feet - bounce and pivot.")

        strike_counts = strike_data.get("strike_counts", {}) if strike_data else {}
        total_hand = sum(strike_counts.get(k, 0) for k in ["jab", "cross", "lead_hook", "rear_hook"])
        total_kicks = sum(strike_counts.get(k, 0) for k in ["front_kick", "roundhouse"])
        if total_hand > 20 and total_kicks < 3:
            suggestions.append("Mix in kicks - they'll stop walking through your punches.")
        if total_kicks > 10 and total_hand < 20:
            suggestions.append("Set up kicks with punches - feint then kick.")

        landed = strike_data.get("total_landed", 0) if strike_data else 0
        thrown = strike_data.get("total_thrown", 0) if strike_data else 0
        if thrown > 5 and landed < 1:
            suggestions.append("Feint more! You're telegraphing your strikes.")
        if thrown > 20 and landed / max(thrown, 1) < 0.2:
            suggestions.append("Stop reaching! Close the distance before striking.")

        if referee_data:
            kd = getattr(referee_data, "knockdown_detected", False) if not isinstance(referee_data, dict) else referee_data.get("knockdown", False)
            if kd:
                suggestions.append("Protect yourself at all times! Recover and survive.")

        if not suggestions:
            suggestions.append("Keep working! Stay focused on the game plan.")

        self.last_suggestions = suggestions[:3]
        return self.last_suggestions

    def check_guard_drop(self, guard_quality):
        if guard_quality == "low":
            return GUARD_SUGGESTIONS["low"]
        elif guard_quality == "mid":
            return GUARD_SUGGESTIONS["mid"]
        return None

    def check_head_movement(self, head_score):
        if head_score < HEAD_MOVEMENT_LOW:
            return "Move your head! Slip and roll."
        return None

    def check_striking_variety(self, strikes, window=50):
        if isinstance(strikes, dict) and "strikes" in strikes:
            strikes = strikes["strikes"]
        if not strikes:
            return None
        types = set(s.get("type", "") for s in strikes[-window:])
        if len(types) < 2:
            return "Mix up your strikes - don't be predictable."
        if "jab" not in types:
            return "Use your jab to set up combinations."
        return None

    def check_pressure(self, movement):
        if isinstance(movement, dict):
            fp = movement.get("forward_pressure", 0)
        else:
            fp = movement.forward_pressure if hasattr(movement, "forward_pressure") else 0
        if fp < FORWARD_PRESSURE_LOW:
            return "Push forward! You're giving ground."
        return None

    def _cooldown_active(self, category, now, cooldown_secs=8.0):
        last = self._last_shown.get(category, 0)
        return (now - last) < cooldown_secs
