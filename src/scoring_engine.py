from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RoundScore:
    round_num: int = 1
    our_landed: int = 0
    our_significant: int = 0
    opponent_landed: int = 0
    opponent_significant: int = 0
    knockdowns_scored: int = 0
    knockdowns_conceded: int = 0
    aggression_score: float = 0.0
    control_time: float = 0.0
    round_grade: str = "10-9"


@dataclass
class MatchScore:
    rounds: list = field(default_factory=list)
    total_our: int = 0
    total_opponent: int = 0
    winner: str = ""
    method: str = "decision"


class ScoringEngine:
    def __init__(self, rounds=3, round_duration=180):
        self.total_rounds = rounds
        self.round_duration = round_duration
        self.round_scores = []
        self.reset_round()

    def reset_round(self, round_num=1):
        self.round_num = round_num
        self.our_strikes_landed = 0
        self.our_significant_strikes = 0
        self.opponent_strikes_landed = 0
        self.opponent_significant_strikes = 0
        self.knockdowns_scored = 0
        self.knockdowns_conceded = 0
        self.aggression_score = 0.0
        self.control_time = 0.0
        self.frame_count = 0

    def update(self, strike_result, movement, referee):
        self.frame_count += 1
        if strike_result:
            self.our_strikes_landed += strike_result.get("total_landed", 0)
            strike_counts = strike_result.get("strike_counts", {}) or {}
            self.our_significant_strikes += sum(
                v for k, v in strike_counts.items()
                if k in ("cross", "rear_hook", "rear_uppercut", "roundhouse", "knee")
            )
        if movement:
            pressure = getattr(movement, "forward_pressure", 0) if not isinstance(movement, dict) else movement.get("forward_pressure", 0)
            self.aggression_score = max(self.aggression_score, abs(pressure))
        if referee:
            if getattr(referee, "knockdown_detected", False):
                self.knockdowns_scored += 1
            if getattr(referee, "stalling_warning", False):
                self.aggression_score = max(self.aggression_score - 0.5, 0)

    def score_round(self, round_num: int) -> RoundScore:
        effective_score = min(self.our_significant_strikes + self.our_strikes_landed * 0.3, 15)
        defense_factor = max(0, 10 - min(self.knockdowns_conceded * 3, 10))
        aggression = min(self.aggression_score * 2, 5)
        total = effective_score * 0.5 + defense_factor + aggression
        if self.knockdowns_scored > 0:
            total += 2
        if total >= 12:
            grade = "10-9"
        elif total >= 10:
            grade = "10-9"
        else:
            grade = "9-9"
        if self.knockdowns_scored >= 3:
            grade = "10-8"
        if self.knockdowns_conceded >= 3:
            grade = "8-10"

        score = RoundScore(
            round_num=round_num,
            our_landed=self.our_strikes_landed,
            our_significant=self.our_significant_strikes,
            opponent_landed=self.opponent_strikes_landed,
            opponent_significant=self.opponent_significant_strikes,
            knockdowns_scored=self.knockdowns_scored,
            knockdowns_conceded=self.knockdowns_conceded,
            aggression_score=round(self.aggression_score, 1),
            control_time=round(self.control_time, 1),
            round_grade=grade,
        )
        self.round_scores.append(score)
        return score

    def score_match(self) -> MatchScore:
        total_our = sum(r.our_landed for r in self.round_scores) if self.round_scores else self.our_strikes_landed
        total_opponent = sum(r.opponent_landed for r in self.round_scores) if self.round_scores else 0
        rounds_won = sum(1 for r in self.round_scores if int(r.round_grade.split("-")[0]) > int(r.round_grade.split("-")[1])) if self.round_scores else 0
        rounds_lost = sum(1 for r in self.round_scores if int(r.round_grade.split("-")[0]) < int(r.round_grade.split("-")[1])) if self.round_scores else 0
        if rounds_won > rounds_lost:
            winner = "us"
        elif rounds_lost > rounds_won:
            winner = "opponent"
        else:
            winner = "draw"
        return MatchScore(
            rounds=list(self.round_scores),
            total_our=total_our,
            total_opponent=total_opponent,
            winner=winner,
            method="decision",
        )


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
        a_wins = sum(1 for r in self.round_scores if getattr(r, "winner", None) == "fighter_a" or (isinstance(r, dict) and r.get("winner") == "fighter_a"))
        b_wins = sum(1 for r in self.round_scores if getattr(r, "winner", None) == "fighter_b" or (isinstance(r, dict) and r.get("winner") == "fighter_b"))

        if a_wins > b_wins:
            self.match_winner = "fighter_a"
            return f"Fighter A wins {a_wins}-{b_wins}"
        elif b_wins > a_wins:
            self.match_winner = "fighter_b"
            return f"Fighter B wins {b_wins}-{a_wins}"
        else:
            return f"Match draw {a_wins}-{b_wins}"
