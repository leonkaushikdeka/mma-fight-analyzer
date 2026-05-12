import cv2
import numpy as np


COLORS = {
    "primary": (0, 255, 255),
    "success": (0, 255, 0),
    "danger": (0, 0, 255),
    "warning": (0, 165, 255),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "dark": (40, 40, 40),
    "dark_bg": (20, 20, 20),
    "accent": (255, 0, 255),
}


class Visualizer:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.strike_flash = 0
        self.last_strike_count = 0

    def render(self, frame, landmarks, movement, strikes, referee, suggestions, score, round_time):
        h, w = frame.shape[:2]
        display = cv2.resize(frame, (self.width, self.height))
        scale_x = self.width / w
        scale_y = self.height / h

        panel_w = 380
        main_w = self.width - panel_w
        main_view = display[:, :main_w].copy()

        self.draw_pose_overlay(main_view, landmarks)
        self.draw_strike_indicators(main_view, strikes)
        self.draw_hud(main_view, movement, score, round_time)

        panel = self._build_panel(movement, strikes, referee, suggestions)

        combined = np.hstack([main_view, panel])
        self.draw_round_timer(combined, round_time)

        return combined

    def draw_hud(self, frame, movement, score, round_time):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 55), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if movement:
            stance = self._get_attr(movement, "stance", "unknown")
            guard = self._get_attr(movement, "guard_quality", "unknown")
            guard_color = COLORS["success"] if guard == "high" else (COLORS["warning"] if guard == "mid" else COLORS["danger"])
            cv2.putText(frame, f"STANCE: {stance.upper()}", (15, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["primary"], 1)
            cv2.putText(frame, f"GUARD: {guard.upper()}", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, guard_color, 1)

        if score:
            grade = self._get_attr(score, "round_grade", "10-9")
            cv2.putText(frame, f"ROUND SCORE: {grade}", (frame.shape[1] - 250, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["primary"], 1)

    def draw_metrics_panel(self, frame, movement):
        panel = np.zeros((200, frame.shape[1], 3), dtype=np.uint8)
        panel[:] = COLORS["dark_bg"]
        if movement:
            pressure = self._get_attr(movement, "forward_pressure", 0)
            head = self._get_attr(movement, "head_movement_score", 0)
            footwork = self._get_attr(movement, "footwork_score", 0)
            cv2.putText(panel, f"Forward Pressure: {pressure:.1f}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["white"], 1)
            cv2.putText(panel, f"Head Movement: {head:.1f}", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["white"], 1)
            cv2.putText(panel, f"Footwork: {footwork:.1f}", (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["white"], 1)
        return panel

    def draw_strike_indicators(self, frame, strikes):
        if not strikes or not isinstance(strikes, dict):
            return
        strike_list = strikes.get("strikes", [])
        for s in strike_list[-3:]:
            stype = s.get("type", "").upper()
            speed = s.get("speed", 0)
            landed = s.get("landed", False)
            color = COLORS["success"] if landed else COLORS["danger"]
            cv2.putText(frame, f"! {stype} {speed:.0f}px/f", (frame.shape[1] - 200, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        total_thrown = strikes.get("total_thrown", 0)
        total_landed = strikes.get("total_landed", 0)
        accuracy = strikes.get("accuracy", 0)
        cv2.putText(frame, f"Strikes: {total_thrown}L/{total_landed} ({accuracy}%)", (15, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["white"], 1)

    def draw_suggestions(self, frame, suggestions):
        if not suggestions:
            return
        overlay = frame.copy()
        y_start = frame.shape[0] - 120
        cv2.rectangle(overlay, (0, y_start), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        for i, sug in enumerate(suggestions[:3]):
            msg = sug if isinstance(sug, str) else getattr(sug, "message", str(sug))
            priority = "medium" if isinstance(sug, str) else getattr(sug, "priority", "medium")
            color_map = {"high": COLORS["danger"], "medium": COLORS["warning"], "low": COLORS["gray"]}
            color = color_map.get(priority, COLORS["white"])
            cv2.putText(frame, f"> {msg}", (15, y_start + 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def draw_pose_overlay(self, frame, landmarks):
        pass

    def draw_round_timer(self, frame, remaining):
        mins, secs = divmod(max(0, int(remaining)), 60)
        timer_text = f"{mins:02d}:{secs:02d}"
        color = COLORS["danger"] if remaining < 30 else (COLORS["warning"] if remaining < 60 else COLORS["success"])
        (tw, th), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
        x = (frame.shape[1] - tw) // 2
        cv2.putText(frame, timer_text, (x, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)

    def draw_frame(self, frame, landmarks, movement_metrics, strike_data, referee_data, suggestions):
        h, w = frame.shape[:2]
        display = cv2.resize(frame, (self.width, self.height))

        panel_w = 380
        main_w = self.width - panel_w
        main_view = display[:, :main_w]

        panel = np.zeros((self.height, panel_w, 3), dtype=np.uint8)
        panel[:] = COLORS["dark_bg"]

        y_offset = 15
        y_offset = self._draw_panel_header(panel, y_offset, "MMA FIGHT ANALYZER")
        y_offset += 5

        if movement_metrics:
            y_offset = self._draw_section(panel, y_offset, "MOVEMENT", COLORS["primary"])
            for key in ["stance", "forward_pressure", "head_movement", "footwork", "guard_position"]:
                val = movement_metrics.get(key, "N/A") if isinstance(movement_metrics, dict) else getattr(movement_metrics, key.replace("position", "quality"), "N/A")
                if isinstance(val, float):
                    y_offset = self._draw_metric(panel, y_offset, key, f"{val:.1f}")
                else:
                    y_offset = self._draw_metric(panel, y_offset, key, str(val))
            y_offset += 5

        if strike_data:
            y_offset = self._draw_section(panel, y_offset, "STRIKES", COLORS["danger"])
            y_offset = self._draw_metric(panel, y_offset, "Thrown", str(strike_data.get("total_thrown", 0)))
            y_offset = self._draw_metric(panel, y_offset, "Landed", str(strike_data.get("total_landed", 0)))
            y_offset = self._draw_metric(panel, y_offset, "Accuracy", f"{strike_data.get('accuracy', 0):.1f}%")
            y_offset = self._draw_metric(panel, y_offset, "Avg Speed", f"{strike_data.get('avg_speed', 0):.1f} px/f")
            y_offset += 5

        if referee_data:
            y_offset = self._draw_section(panel, y_offset, "REFEREE", COLORS["warning"])
            kd = referee_data.get("knockdown", False) if isinstance(referee_data, dict) else getattr(referee_data, "knockdown_detected", False)
            if kd:
                cv2.putText(panel, "KNOCKDOWN!", (20, y_offset),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, COLORS["danger"], 2)
                y_offset += 35
            su = referee_data.get("standup", False) if isinstance(referee_data, dict) else getattr(referee_data, "standup_detected", False)
            if su:
                cv2.putText(panel, "STANDUP!", (20, y_offset),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, COLORS["success"], 2)
                y_offset += 35

            standing = referee_data.get("standing", True) if isinstance(referee_data, dict) else True
            status_color = COLORS["success"] if standing else COLORS["danger"]
            status_text = "STANDING" if standing else "DOWN"
            y_offset = self._draw_metric(panel, y_offset, "Status", status_text, value_color=status_color)

            commentary = referee_data.get("commentary", "") if isinstance(referee_data, dict) else getattr(referee_data, "illegal_strike", "")
            if commentary:
                cv2.putText(panel, f'"{commentary}"', (15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["warning"], 1)
                y_offset += 30

            warnings = referee_data.get("warnings", []) if isinstance(referee_data, dict) else []
            if warnings:
                cv2.putText(panel, f'Warnings: {len(warnings)}', (15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["danger"], 1)
                y_offset += 25

            y_offset += 5

        if suggestions:
            y_offset = self._draw_section(panel, y_offset, "SUGGESTIONS", COLORS["accent"])
            for i, sug in enumerate(suggestions[:3]):
                msg = sug if isinstance(sug, str) else getattr(sug, "message", str(sug))
                cv2.putText(panel, f"  {i+1}. {msg}", (15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["white"], 1)
                y_offset += 22

        combined = np.hstack([main_view, panel])
        return combined

    def draw_pose_debug(self, frame, landmarks, movement_metrics=None):
        return frame

    def _build_panel(self, movement, strikes, referee, suggestions):
        panel = np.zeros((self.height, 380, 3), dtype=np.uint8)
        panel[:] = COLORS["dark_bg"]
        y_offset = 15
        y_offset = self._draw_panel_header(panel, y_offset, "MMA FIGHT ANALYZER")
        y_offset += 5

        if movement:
            y_offset = self._draw_section(panel, y_offset, "MOVEMENT", COLORS["primary"])
            for key, label, fmt in [
                ("stance", "Stance", "str"),
                ("forward_pressure", "Pressure", "float"),
                ("head_movement_score", "Head Move", "float"),
                ("footwork_score", "Footwork", "float"),
                ("guard_quality", "Guard", "str"),
            ]:
                val = self._get_attr(movement, key, "N/A")
                if fmt == "float":
                    y_offset = self._draw_metric(panel, y_offset, label, f"{val:.1f}" if isinstance(val, float) else str(val))
                else:
                    y_offset = self._draw_metric(panel, y_offset, label, str(val))
            y_offset += 5

        if strikes:
            y_offset = self._draw_section(panel, y_offset, "STRIKES", COLORS["danger"])
            for label, key in [("Thrown", "total_thrown"), ("Landed", "total_landed"), ("Accuracy", "accuracy"), ("Avg Speed", "avg_speed")]:
                val = strikes.get(key, 0)
                if "Accuracy" in label or "Speed" in label:
                    y_offset = self._draw_metric(panel, y_offset, label, f"{val:.1f}{'%' if 'Acc' in label else ' px/f'}")
                else:
                    y_offset = self._draw_metric(panel, y_offset, label, str(val))
            y_offset += 5

        if referee:
            y_offset = self._draw_section(panel, y_offset, "REFEREE", COLORS["warning"])
            kd = self._get_attr(referee, "knockdown_detected", False)
            if kd:
                cv2.putText(panel, "KNOCKDOWN!", (20, y_offset), cv2.FONT_HERSHEY_DUPLEX, 1.0, COLORS["danger"], 2)
                y_offset += 35
            msg = self._get_attr(referee, "illegal_strike", "")
            if msg:
                cv2.putText(panel, f'"{msg}"', (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["warning"], 1)
                y_offset += 30
            y_offset += 5

        if suggestions:
            y_offset = self._draw_section(panel, y_offset, "SUGGESTIONS", COLORS["accent"])
            for i, sug in enumerate(suggestions[:3]):
                msg = sug if isinstance(sug, str) else getattr(sug, "message", str(sug))
                cv2.putText(panel, f"  {i+1}. {msg}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["white"], 1)
                y_offset += 22

        return panel

    def _draw_panel_header(self, panel, y, text):
        cv2.rectangle(panel, (0, y), (panel.shape[1], y + 40), COLORS["dark"], -1)
        cv2.putText(panel, text, (20, y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, COLORS["primary"], 1)
        cv2.line(panel, (10, y + 42), (panel.shape[1] - 10, y + 42), COLORS["gray"], 1)
        return y + 48

    def _draw_section(self, panel, y, title, color):
        cv2.putText(panel, title, (15, y), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
        cv2.line(panel, (15, y + 4), (panel.shape[1] - 15, y + 4), color, 1)
        return y + 18

    def _draw_metric(self, panel, y, label, value, value_color=None):
        cv2.putText(panel, f"  {label.replace('_', ' ').title()}:", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["gray"], 1)
        color = value_color or COLORS["white"]
        cv2.putText(panel, str(value), (panel.shape[1] - 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return y + 22

    def add_round_overlay(self, frame, round_num):
        cv2.putText(frame, f"ROUND {round_num}", (self.width // 2 - 80, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, COLORS["white"], 2)
        return frame

    @staticmethod
    def _get_attr(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
