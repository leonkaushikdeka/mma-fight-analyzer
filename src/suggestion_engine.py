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

    def generate(self, movement_data, strike_data, referee_data, elapsed_seconds):
        if self.suggestion_cooldown > 0:
            self.suggestion_cooldown -= 1
            return []
        self.suggestion_cooldown = 3

        suggestions = []

        guard = movement_data.get("guard_position", "unknown")
        if guard in GUARD_SUGGESTIONS:
            suggestions.append(GUARD_SUGGESTIONS[guard])

        stance = movement_data.get("stance", "unknown")
        if "unknown" in stance:
            suggestions.append(STANCE_SUGGESTIONS["unknown"])

        head_movement = movement_data.get("head_movement", 0)
        if head_movement < HEAD_MOVEMENT_LOW:
            suggestions.append("Move your head! Slip and roll after combinations.")
        elif head_movement > HEAD_MOVEMENT_HIGH:
            suggestions.append("Great head movement - keep varying the rhythm.")

        forward_pressure = movement_data.get("forward_pressure", 0)
        if forward_pressure > FORWARD_PRESSURE_LOW * 3:
            suggestions.append("Cut off the cage - don't let them circle out.")
        elif forward_pressure < FORWARD_PRESSURE_LOW:
            suggestions.append("Push forward! You're giving ground.")

        footwork = movement_data.get("footwork", 0)
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

        if referee_data and referee_data.get("knockdown"):
            suggestions.append("Protect yourself at all times! Recover and survive.")

        if not suggestions:
            suggestions.append("Keep working! Stay focused on the game plan.")

        self.last_suggestions = suggestions[:3]
        return self.last_suggestions
