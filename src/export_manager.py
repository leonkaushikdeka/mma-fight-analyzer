import csv
import os
import time
from collections import deque


class ExportManager:
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.events = deque(maxlen=10000)
        self.session_start = time.time()
        self.fight_log = None
        self.round_logs = {}
        self._file = None
        self._writer = None

    def start_fight(self, filename=None):
        if filename is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"mma_fight_{ts}.csv"
        path = os.path.join(self.output_dir, filename)
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "timestamp_s", "event_type", "fighter", "value",
            "round", "detail",
        ])
        self._file.flush()
        print(f"Exporting to {path}")
        return path

    def log_event(self, event_type, fighter="A", value="", round_num=1, detail=""):
        now = time.time() - self.session_start
        row = [
            round(now, 2),
            event_type,
            fighter,
            str(value),
            round_num,
            detail,
        ]
        self.events.append(row)
        if self._writer:
            self._writer.writerow(row)
            self._file.flush()

    def log_strike(self, strike_info, fighter="A", round_num=1):
        self.log_event(
            event_type=f"strike_{strike_info.get('type', 'unknown')}",
            fighter=fighter,
            value=strike_info.get("speed", 0),
            round_num=round_num,
            detail=f"side={strike_info.get('side','?')}|landed={strike_info.get('landed',False)}",
        )

    def log_knockdown(self, fighter="A", round_num=1):
        self.log_event("knockdown", fighter, "", round_num)

    def log_takedown(self, fighter="A", round_num=1):
        self.log_event("takedown", fighter, "", round_num)

    def log_round_score(self, round_num, score_data):
        self.log_event("round_score", "A", str(score_data), round_num)

    def end_fight(self, summary=None):
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

        rs = time.strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.output_dir, f"mma_summary_{rs}.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_s", "event_type", "fighter", "value", "round", "detail"])
            for row in self.events:
                w.writerow(row)
            if summary:
                w.writerow([])
                w.writerow(["SUMMARY"])
                for k, v in summary.items():
                    w.writerow([k, str(v)])

        return summary_path
