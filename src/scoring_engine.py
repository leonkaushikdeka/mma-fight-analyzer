class ScoringEngine:
    def __init__(self):
        self.reset_round()

    def reset_round(self, round_num=1):
        self.round_num = round_num
        self.effective_striking = {"fighter_a": 0, "fighter_b": 0}
        self.grappling_control = {"fighter_a": 0, "fighter_b": 0}
        self.aggression = {"fighter_a": 0, "fighter_b": 0}
        self.cage_control = {"fighter_a": 0, "fighter_b": 0}
        self.takedowns = {"fighter_a": 0, "fighter_b": 0}
        self.knockdowns = {"fighter_a": 0, "fighter_b": 0}
        self.significant_strikes = {"fighter_a": 0, "fighter_b": 0}
        self.total_strikes = {"fighter_a": 0, "fighter_b": 0}
        self.defense_rating = {"fighter_a": 0, "fighter_b": 0}

    def score_round(self, strike_data, movement_data, knockdowns=None):
        result = {
            "round": self.round_num,
            "scores": {},
            "winner": None,
            "analysis": [],
        }

        effective_striking_a = strike_data.get("total_landed", 0) if strike_data else 0
        effective_striking_b = 0
        result["significant_strikes"] = {
            "fighter_a": effective_striking_a,
            "fighter_b": effective_striking_b,
        }

        stance_a = movement_data.get("stance", "unknown") if movement_data else "unknown"
        forward_pressure_a = movement_data.get("forward_pressure", 0) if movement_data else 0
        head_movement_a = movement_data.get("head_movement", 0) if movement_data else 0

        self.aggression = {"fighter_a": min(forward_pressure_a * 5, 10), "fighter_b": 0}

        striking_score_a = min(effective_striking_a, 10)
        striking_score_b = 0

        result["scores"]["fighter_a"] = {
            "effective_striking": round(striking_score_a, 1),
            "aggression": round(self.aggression["fighter_a"], 1),
            "total": round(striking_score_a + self.aggression["fighter_a"], 1),
        }
        result["scores"]["fighter_b"] = {
            "effective_striking": round(striking_score_b, 1),
            "aggression": round(self.aggression["fighter_b"], 1),
            "total": round(striking_score_b + self.aggression["fighter_b"], 1),
        }

        total_a = result["scores"]["fighter_a"]["total"]
        total_b = result["scores"]["fighter_b"]["total"]

        if total_a > total_b + 2:
            result["winner"] = "fighter_a"
            result["analysis"].append(f"Fighter A clearly wins round {self.round_num} "
                                      f"({total_a:.0f}-{total_b:.0f})")
        elif total_a > total_b:
            result["winner"] = "fighter_a"
            result["analysis"].append(f"Fighter A edges round {self.round_num} "
                                      f"({total_a:.0f}-{total_b:.0f})")
        else:
            result["winner"] = "draw"
            result["analysis"].append(f"Round {self.round_num} is close ({total_a:.0f}-{total_b:.0f})")

        if stance_a:
            result["analysis"].append(f"Fighter A fighting from {stance_a} stance")

        return result

    def get_round_summary(self):
        return {
            "round": self.round_num,
            "total_strikes": self.total_strikes,
            "significant_strikes": self.significant_strikes,
            "takedowns": self.takedowns,
            "knockdowns": self.knockdowns,
        }


class MatchManager:
    def __init__(self, rounds=3):
        self.rounds = rounds
        self.current_round = 1
        self.round_scores = []
        self.match_winner = None

    def next_round(self):
        if self.current_round < self.rounds:
            self.current_round += 1
            return True
        return False

    def record_round_score(self, score_data):
        self.round_scores.append(score_data)

    def get_match_result(self):
        a_wins = sum(1 for r in self.round_scores if r.get("winner") == "fighter_a")
        b_wins = sum(1 for r in self.round_scores if r.get("winner") == "fighter_b")

        if a_wins > b_wins:
            self.match_winner = "fighter_a"
            return f"Fighter A wins {a_wins}-{b_wins}"
        elif b_wins > a_wins:
            self.match_winner = "fighter_b"
            return f"Fighter B wins {b_wins}-{a_wins}"
        else:
            return f"Match draw {a_wins}-{b_wins}"
