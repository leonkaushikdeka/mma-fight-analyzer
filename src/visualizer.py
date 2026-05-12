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
        self.overlay = None

    def draw_frame(self, frame, landmarks, movement_metrics, strike_data, referee_data, suggestions):
        h, w = frame.shape[:2]
        display = cv2.resize(frame, (self.width, self.height))
        scale_x = self.width / w
        scale_y = self.height / h

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
            for key, val in movement_metrics.items():
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
            if referee_data.get("knockdown"):
                cv2.putText(panel, "KNOCKDOWN!", (20, y_offset),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, COLORS["danger"], 2)
                y_offset += 35
            elif referee_data.get("standup"):
                cv2.putText(panel, "STANDUP!", (20, y_offset),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, COLORS["success"], 2)
                y_offset += 35

            standing = referee_data.get("standing", True)
            status_color = COLORS["success"] if standing else COLORS["danger"]
            status_text = "STANDING" if standing else "DOWN"
            y_offset = self._draw_metric(panel, y_offset, "Status", status_text, value_color=status_color)

            commentary = referee_data.get("commentary", "")
            if commentary:
                cv2.putText(panel, f'"{commentary}"', (15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["warning"], 1)
                y_offset += 30

            warnings = referee_data.get("warnings", [])
            if warnings:
                cv2.putText(panel, f'Warnings: {len(warnings)}', (15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["danger"], 1)
                y_offset += 25

            y_offset += 5

        if suggestions:
            y_offset = self._draw_section(panel, y_offset, "SUGGESTIONS", COLORS["accent"])
            for i, sug in enumerate(suggestions[:3]):
                cv2.putText(panel, f"  {i+1}. {sug}", (15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["white"], 1)
                y_offset += 22

        combined = np.hstack([main_view, panel])
        return combined

    def draw_pose_debug(self, frame, landmarks, movement_metrics=None):
        if movement_metrics:
            guard = movement_metrics.get("guard_position", "")
            guard_color = COLORS["success"] if guard == "high" else (COLORS["warning"] if guard == "mid" else COLORS["danger"])
            cv2.putText(frame, f"GUARD: {guard.upper()}", (30, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, guard_color, 2)
        return frame

    def _draw_panel_header(self, panel, y, text):
        cv2.rectangle(panel, (0, y), (self.width, y + 40), COLORS["dark"], -1)
        cv2.putText(panel, text, (20, y + 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, COLORS["primary"], 1)
        cv2.line(panel, (10, y + 42), (panel.shape[1] - 10, y + 42), COLORS["gray"], 1)
        return y + 48

    def _draw_section(self, panel, y, title, color):
        cv2.putText(panel, title, (15, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
        cv2.line(panel, (15, y + 4), (panel.shape[1] - 15, y + 4), color, 1)
        return y + 18

    def _draw_metric(self, panel, y, label, value, value_color=None):
        cv2.putText(panel, f"  {label.replace('_', ' ').title()}:", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["gray"], 1)
        color = value_color or COLORS["white"]
        cv2.putText(panel, str(value), (panel.shape[1] - 120, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return y + 22

    def add_round_overlay(self, frame, round_num):
        cv2.putText(frame, f"ROUND {round_num}", (self.width // 2 - 80, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, COLORS["white"], 2)
        return frame
