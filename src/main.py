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
    def __init__(self, source=0, width=1280, height=720, fps=30, skip_frames=0):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.skip_frames = skip_frames

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
        self.processed_count = 0
        self.start_time = None
        self.round_start_time = None
        self.suggestion_timer = 0

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Error: Cannot open video source {self.source}")
            return

        if isinstance(self.source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Webcam: {actual_w}x{actual_h} @ ~{self.fps}fps")

        self.running = True
        self.start_time = time.time()
        self.round_start_time = time.time()

        print("MMA Fight Analyzer running...")
        print("Controls: [q] quit | [p] pause | [r] reset round | [SPACE] toggle overlay | [+] fps [+] | [-] fps [-]")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break

                self.frame_count += 1
                elapsed = time.time() - self.start_time

                if self.paused:
                    cv2.waitKey(100)
                    continue

                if self.skip_frames > 0 and self.frame_count % (self.skip_frames + 1) != 0:
                    display = frame
                    display = cv2.resize(display, (self.width, self.height))
                    self._draw_minimal_hud(display, elapsed)
                    cv2.imshow("MMA Fight Analyzer", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                    elif key == ord('p'):
                        self.paused = not self.paused
                    continue

                self.processed_count += 1

                landmarks_obj = None
                landmarks_arr = None
                try:
                    result = self.pose_estimator.process_frame(frame)
                    if result and result.pose_landmarks:
                        landmarks_obj = result.pose_landmarks
                        landmarks_arr = self.pose_estimator.landmarks_to_array(
                            landmarks_obj, frame.shape
                        )
                except Exception as e:
                    print(f"Pose error: {e}")
                    continue

                move_data = self.movement.analyze(landmarks_arr)
                strike_data = self.strikes.analyze(landmarks_arr)
                ref_data = self.referee.analyze(
                    landmarks_arr,
                    strike_data.get("strikes", []) if strike_data else []
                )

                self.suggestion_timer += 1
                if self.suggestion_timer % 5 == 0:
                    suggestions = self.suggestions.generate(
                        move_data, strike_data, ref_data, elapsed
                    )
                else:
                    suggestions = []

                display_frame = frame.copy()
                if landmarks_obj:
                    self.pose_estimator.draw_landmarks(display_frame, landmarks_obj)

                display = self.viz.draw_frame(
                    display_frame, landmarks_arr, move_data,
                    strike_data, ref_data, suggestions
                )

                cv2.imshow("MMA Fight Analyzer", display)
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Runtime error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose_estimator.close()
            self._print_summary(time.time() - self.start_time)

    def _handle_key(self, key):
        if key == ord('q'):
            self.running = False
            return False
        elif key == ord('p'):
            self.paused = not self.paused
            print(f"{'Paused' if self.paused else 'Resumed'}")
        elif key == ord('r'):
            self.scoring.reset_round(self.match.current_round)
            self.referee.reset_round()
            self.round_start_time = time.time()
            print(f"Reset round {self.match.current_round}")
        elif key == ord('+') or key == ord('='):
            self.skip_frames = min(self.skip_frames + 1, 5)
            print(f"Skip frames: {self.skip_frames}")
        elif key == ord('-') or key == ord('_'):
            self.skip_frames = max(self.skip_frames - 1, 0)
            print(f"Skip frames: {self.skip_frames}")
        return True

    def _draw_minimal_hud(self, frame, elapsed):
        cv2.putText(frame, f"MMA Analyzer | Frame {self.frame_count} | {elapsed:.0f}s | Skipping...",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    def _print_summary(self, elapsed):
        print("\n" + "=" * 55)
        print("  MATCH SUMMARY")
        print("=" * 55)
        print(f"  Duration:          {int(elapsed // 60)}m {int(elapsed % 60)}s")
        print(f"  Frames captured:   {self.frame_count}")
        print(f"  Frames processed:  {self.processed_count}")
        print(f"  Strikes thrown:    {self.strikes.strikes_thrown}")
        print(f"  Strikes landed:    {self.strikes.strikes_landed}")
        thrown = self.strikes.strikes_thrown
        landed = self.strikes.strikes_landed
        print(f"  Accuracy:          {landed / max(thrown, 1) * 100:.1f}%")
        print(f"  Knockdowns:        {self.referee.knockdowns}")
        speeds = self.strikes.strike_speeds
        print(f"  Avg strike speed:  {np.mean(speeds) if speeds else 0:.1f} px/s")
        if self.strikes.strike_counts:
            active = {k: v for k, v in sorted(self.strikes.strike_counts.items(), key=lambda x: -x[1]) if v > 0}
            if active:
                print(f"  Strike mix:        {active}")
        print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="MMA Fight Analyzer - Real-time strike detection, referee AI & coaching"
    )
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: 0 for webcam, or path to video file")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30,
                        help="Nominal framerate for time calculations")
    parser.add_argument("--skip", type=int, default=0, metavar="N",
                        help="Process every Nth+1 frame for perf. 0=all, 1=every other")
    parser.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                        help="MediaPipe model complexity: 0=lite, 1=full, 2=heavy")

    args = parser.parse_args()

    print(r"""
    __  __ ___   ___ _   _ ___ ___ ___  _  _   _   ___
   |  \/  / __| / __| | | / __| __| _ \| \| | /_\ | _ \_ __
   | |\/| \__ \ \__ \ |_| \__ \ _||   /| .` |/ _ \|  _/| '_ \
   |_|  |_|___/ |___/\___/|___/_| |_|_\_|\_/_/ \_\_|  | .__/
                                                         |_|
    Real-time MMA analysis, strike detection & coaching AI (offline)
    """)

    try:
        source_int = int(args.source)
    except ValueError:
        source_int = args.source

    analyzer = MMAFightAnalyzer(
        source=source_int,
        width=args.width,
        height=args.height,
        fps=args.fps,
        skip_frames=args.skip,
    )
    analyzer.run()


if __name__ == "__main__":
    main()
