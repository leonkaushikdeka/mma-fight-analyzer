import cv2
import numpy as np
import time


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
    "blue": (255, 150, 0),
}


class Visualizer:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.last_frame_time = time.perf_counter()
        self.fps_history = []

    def draw_frame(self, frame, landmarks, movement_metrics, strike_data, referee_data, suggestions):
        now = time.perf_counter()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if 0.001 < dt < 1.0:
            self.fps_history.append(1.0 / dt)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
        fps_display = round(np.mean(self.fps_history)) if self.fps_history else 0

        h, w = frame.shape[:2]
        display = cv2.resize(frame, (self.width, self.height)) if w != self.width else frame.copy()
        actual_h, actual_w = display.shape[:2]

        panel_w = 400 if self.width >= 1280 else int(self.width * 0.28)
        main_w = actual_w - panel_w
        if main_w < 400:
            panel_w = max(200, actual_w - 400)
            main_w = actual_w - panel_w

        main_view = display[:, :main_w]
        panel = np.zeros((actual_h, panel_w, 3), dtype=np.uint8)
        panel[:] = COLORS["dark_bg"]

        y = 12
        y = self._header(panel, y, "MMA ANALYZER", COLORS["primary"])
        y += 2

        if movement_metrics:
            y = self._section(panel, y, "MOVEMENT", COLORS["primary"])
            for k, v in movement_metrics.items():
                if k == "body_scale":
                    continue
                label = k.replace("_", " ").title()
                color = COLORS["white"]
                if k == "guard_position":
                    color = COLORS["success"] if v == "high" else (COLORS["warning"] if v == "mid" else COLORS["danger"])
                y = self._row(panel, y, label, str(v), color)
            y += 4

        if strike_data:
            y = self._section(panel, y, "STRIKES", COLORS["danger"])
            y = self._row(panel, y, "Thrown", str(strike_data.get("total_thrown", 0)))
            y = self._row(panel, y, "Landed", str(strike_data.get("total_landed", 0)))
            landed = strike_data.get("total_landed", 0)
            thrown = strike_data.get("total_thrown", 0)
            acc = strike_data.get("accuracy", round(landed / max(thrown, 1) * 100, 1))
            y = self._row(panel, y, "Accuracy", f"{acc}%",
                          COLORS["success"] if acc > 30 else (COLORS["warning"] if acc > 15 else COLORS["danger"]))
            y = self._row(panel, y, "Avg Speed", f'{strike_data.get("avg_speed", 0):.1f}')

            sc = strike_data.get("strike_counts", {})
            active = {k: v for k, v in sorted(sc.items(), key=lambda x: -x[1]) if v > 0}
            if active:
                breakdown = " ".join(f"{k[:4]}={v}" for k, v in list(active.items())[:4])
                y = self._row(panel, y, "Breakdown", breakdown, COLORS["gray"])
            y += 4

        if referee_data:
            y = self._section(panel, y, "REFEREE", COLORS["warning"])
            state = referee_data.get("state", "standing")
            state_color = COLORS["success"] if state == "standing" else COLORS["danger"]
            y = self._row(panel, y, "Status", state.upper(), state_color)

            if referee_data.get("knockdown"):
                box = np.zeros((35, panel_w - 20, 3), dtype=np.uint8)
                box[:] = (0, 0, 80)
                cv2.putText(box, "!! KNOCKDOWN !!", (10, 25),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, COLORS["danger"], 2)
                panel[y:y + 35, 10:panel_w - 10] = box
                y += 40
            elif referee_data.get("standup"):
                y = self._row(panel, y, "STANDUP!", "", COLORS["success"])
                y += 3

            commentary = referee_data.get("commentary", "")
            if commentary:
                cv2.putText(panel, f' "{commentary}"', (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["warning"], 1)
                y += 24
            y += 4

        if suggestions:
            y = self._section(panel, y, "COACHING", COLORS["accent"])
            for i, sug in enumerate(suggestions[:3], 1):
                cv2.putText(panel, f" {i}. {sug}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["white"], 1)
                y += 20

        cv2.putText(panel, f"FPS: {fps_display}", (panel_w - 100, actual_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["success"], 1)

        separator = np.zeros((actual_h, 3, 3), dtype=np.uint8)
        separator[:] = COLORS["gray"]
        combined = np.hstack([main_view, separator, panel])
        return combined

    def _header(self, panel, y, text, color):
        cv2.rectangle(panel, (0, y), (panel.shape[1], y + 35), COLORS["dark"], -1)
        cv2.putText(panel, text, (15, y + 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1)
        cv2.line(panel, (8, y + 38), (panel.shape[1] - 8, y + 38), COLORS["gray"], 1)
        return y + 42

    def _section(self, panel, y, title, color):
        cv2.putText(panel, title, (12, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1)
        cv2.line(panel, (12, y + 4), (panel.shape[1] - 12, y + 4), color, 1)
        return y + 16

    def _row(self, panel, y, label, value, value_color=None):
        cv2.putText(panel, f" {label}:", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["gray"], 1)
        if value:
            color = value_color or COLORS["white"]
            cv2.putText(panel, str(value), (panel.shape[1] - 130, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return y + 20
