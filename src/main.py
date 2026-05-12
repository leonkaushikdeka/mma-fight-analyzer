import cv2
import numpy as np
import argparse
import time
import sys
import os

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

        self.pose_estimator = PoseEstimator(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.movement = MovementAnalyzer(history_frames=15, fps=fps)
        self.strikes = StrikeDetector(fps=fps)
        self.scoring = ScoringEngine()
        self.match = MatchManager(rounds=3)
        self.referee = RefereeAI(fps=fps)
        self.suggestions = SuggestionEngine()
        self.viz = Visualizer(width=width, height=height)

        self.running = False
        self.paused = False
        self.frame_count = 0
        self.start_time = None
        self.round_start_time = None

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
        print("Controls: [q] quit  [p] pause  [r] reset round  [SPACE] toggle overlay")
        show_overlay = True

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break

                if self.paused:
                    key = cv2.waitKey(100) & 0xFF
                    self._handle_key(key)
                    continue

                self.frame_count += 1
                elapsed = time.time() - self.start_time

                landmarks_obj = None
                landmarks_arr = None
                result = self.pose_estimator.process_frame(frame)
                if result and result.pose_landmarks:
                    landmarks_obj = result.pose_landmarks
                    landmarks_arr = self.pose_estimator.landmarks_to_array(
                        landmarks_obj, frame.shape
                    )
                    self.pose_estimator.draw_landmarks(frame, landmarks_obj)

                move_data = self.movement.analyze(landmarks_arr)
                strike_data = self.strikes.analyze(landmarks_arr)
                ref_data = self.referee.analyze(landmarks_arr, strike_data.get("strikes", []) if strike_data else [])
                suggestions = self.suggestions.generate(
                    move_data, strike_data, ref_data, elapsed
                )

                if show_overlay:
                    display = self.viz.draw_frame(
                        frame, landmarks_arr, move_data,
                        strike_data, ref_data, suggestions
                    )
                else:
                    display = frame

                self._draw_status_bar(display, elapsed)
                self._handle_key(cv2.waitKey(1) & 0xFF)

                cv2.imshow("MMA Fight Analyzer", display)

        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose_estimator.close()
            self._print_summary(elapsed if 'elapsed' in dir() else 0)

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

    def _draw_status_bar(self, frame, elapsed):
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
        print(f"Accuracy: {self.strikes.strikes_landed / max(self.strikes.strikes_thrown, 1) * 100:.1f}%")
        print(f"Knockdowns: {self.referee.knockdowns}")
        print(f"Average strike speed: {np.mean(self.strikes.strike_speeds) if self.strikes.strike_speeds else 0:.1f}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="MMA Fight Analyzer - Real-time pose analysis for MMA training"
    )
    parser.add_argument("--source", type=int, default=0,
                        help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Display width")
    parser.add_argument("--height", type=int, default=720,
                        help="Display height")
    parser.add_argument("--fps", type=int, default=30,
                        help="Processing framerate")

    args = parser.parse_args()

    print(r"""
    __  __ ___   ___ _   _ ___ ___ ___  _  _   _   ___
   |  \/  / __| / __| | | / __| __| _ \| \| | /_\ | _ \_ __
   | |\/| \__ \ \__ \ |_| \__ \ _||   /| .` |/ _ \|  _/| '_ \
   |_|  |_|___/ |___/\___/|___/_| |_|_\_|_|\_/_/ \_\_|  | .__/
                                                         |_|
    Real-time MMA analysis, strike detection & coaching AI
    """)

    analyzer = MMAFightAnalyzer(
        source=args.source,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    analyzer.run()


if __name__ == "__main__":
    main()
