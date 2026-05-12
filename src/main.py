import cv2
import numpy as np
import argparse
import time
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pose_estimation import PoseEstimator
from src.movement_analyzer import MovementAnalyzer
from src.strike_detector import StrikeDetector
from src.scoring_engine import ScoringEngine, MatchManager
from src.referee_ai import RefereeAI
from src.suggestion_engine import SuggestionEngine
from src.visualizer import Visualizer


class MMAFightAnalyzer:
    def __init__(self, source=0, width=1280, height=720, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.round_duration = 180

        self.pose_estimator = PoseEstimator(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.movement = MovementAnalyzer(history_frames=15, fps=fps)
        self.strikes = StrikeDetector(fps=fps)
        self.scoring = ScoringEngine(rounds=3, round_duration=self.round_duration)
        self.match = MatchManager(rounds=3)
        self.referee = RefereeAI(fps=fps)
        self.suggestions = SuggestionEngine()
        self.viz = Visualizer(width=width, height=height)

        self.running = False
        self.paused = False
        self.frame_count = 0
        self.start_time = None
        self.round_start_time = None
        self.overlay_enabled = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Error: Cannot open video source {self.source}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.running = True
        self.start_time = time.time()
        self.round_start_time = time.time()

        print("MMA Fight Analyzer running...")
        print("Controls: [q] quit  [p] pause  [r] reset round  [s] screenshot  [o] overlay toggle")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break

                if self.paused:
                    cv2.imshow("MMA Fight Analyzer", frame)
                    key = cv2.waitKey(100) & 0xFF
                    self._handle_key(key)
                    continue

                self.frame_count += 1
                elapsed = time.time() - self.start_time
                round_time = self.round_duration - (time.time() - self.round_start_time)

                # 1. Pose Estimation
                pose_result, mp_result = self.pose_estimator.process_frame(frame)
                landmarks_arr = pose_result.landmarks if pose_result.has_detection else None

                if pose_result.has_detection:
                    self.pose_estimator.draw_landmarks(frame, mp_result.pose_landmarks)

                # 2. Movement Analysis
                move_metrics = self.movement.analyze(landmarks_arr)

                # 3. Strike Detection
                strike_data = self.strikes.analyze(landmarks_arr)

                # 4. Referee AI
                ref_result = self.referee.analyze(
                    landmarks_arr,
                    strike_data.get("strikes", []) if strike_data else []
                )

                # 5. Suggestion Engine
                suggestions = self.suggestions.generate(
                    move_metrics, strike_data, ref_result, elapsed
                )

                # 6. Scoring
                self.scoring.update(strike_data, move_metrics, ref_result)
                round_score = self.scoring.score_round(self.match.current_round)

                # 7. Visualization
                if self.overlay_enabled:
                    display = self.viz.render(
                        frame, landmarks_arr, move_metrics,
                        strike_data, ref_result, suggestions,
                        round_score, round_time
                    )
                else:
                    display = frame

                self._draw_status_bar(display, elapsed)

                cv2.imshow("MMA Fight Analyzer", display)
                self._handle_key(cv2.waitKey(1) & 0xFF)

        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose_estimator.close()
            self._print_summary(elapsed if 'elapsed' in locals() else 0)

    def _handle_key(self, key):
        if key == ord('q'):
            self.running = False
        elif key == ord('p'):
            self.paused = not self.paused
            print(f"{'Paused' if self.paused else 'Resumed'}")
        elif key == ord('r'):
            self.scoring.reset_round(self.match.current_round)
            self.referee.reset_round()
            self.round_start_time = time.time()
            print(f"Reset round {self.match.current_round}")
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mma_screenshot_{timestamp}.png"
            cv2.imwrite(filename, self._last_frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('o'):
            self.overlay_enabled = not self.overlay_enabled

    def _draw_status_bar(self, frame, elapsed):
        self._last_frame = frame.copy()
        bar = np.zeros((30, frame.shape[1], 3), dtype=np.uint8)
        bar[:] = (20, 20, 20)
        mins, secs = divmod(int(elapsed), 60)
        round_secs = int(time.time() - self.round_start_time)
        r_mins, r_secs = divmod(round_secs, 60)

        cv2.putText(bar, f"TIME: {mins:02d}:{secs:02d}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(bar, f"ROUND {self.match.current_round}: {r_mins:02d}:{r_secs:02d}",
                    (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(bar, f"FPS: {self.frame_count / max(elapsed, 0.1):.0f}",
                    (frame.shape[1] - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        frame[-30:, :] = bar

    def _print_summary(self, elapsed):
        print("\n" + "=" * 50)
        print("MATCH SUMMARY")
        print("=" * 50)
        print(f"Duration: {int(elapsed // 60)}m {int(elapsed % 60)}s")
        print(f"Frames processed: {self.frame_count}")
        print(f"Strikes thrown: {self.strikes.strikes_thrown}")
        print(f"Strikes landed: {self.strikes.strikes_landed}")
        acc = self.strikes.strikes_landed / max(self.strikes.strikes_thrown, 1) * 100
        print(f"Accuracy: {acc:.1f}%")
        print(f"Knockdowns: {self.referee.knockdowns}")
        avg_speed = np.mean(self.strikes.strike_speeds) if self.strikes.strike_speeds else 0
        print(f"Average strike speed: {avg_speed:.1f}")
        match_result = self.match.get_match_result()
        print(f"Match result: {match_result}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="MMA Fight Analyzer - Real-time pose analysis for MMA training"
    )
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Display width")
    parser.add_argument("--height", type=int, default=720,
                        help="Display height")
    parser.add_argument("--fps", type=int, default=30,
                        help="Processing framerate")

    args = parser.parse_args()

    source = 0 if args.source == "0" else args.source

    print(r"""
    __  __ ___   ___ _   _ ___ ___ ___  _  _   _   ___
   |  \/  / __| / __| | | / __| __| _ \| \| | /_\ | _ \_ __
   | |\/| \__ \ \__ \ |_| \__ \ _||   /| .` |/ _ \|  _/| '_ \
   |_|  |_|___/ |___/\___/|___/_| |_|_\_|_|\_/_/ \_\_|  | .__/
                                                         |_|
    Real-time MMA analysis, strike detection & coaching AI
    """)

    analyzer = MMAFightAnalyzer(
        source=source,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    analyzer.run()


if __name__ == "__main__":
    main()
