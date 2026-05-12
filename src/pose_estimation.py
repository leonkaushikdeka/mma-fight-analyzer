import cv2
import numpy as np
import mediapipe as mp
import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PoseResult:
    landmarks: Optional[np.ndarray]
    timestamp: float
    frame_width: int
    frame_height: int
    has_detection: bool


LANDMARK_INDICES = {
    "nose": 0,
    "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8,
    "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}


class PoseEstimator:
    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

    def process_frame(self, frame: np.ndarray) -> Tuple[PoseResult, any]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_result = self.pose.process(rgb)
        rgb.flags.writeable = True
        landmarks_arr = None
        has_detection = False
        if mp_result and mp_result.pose_landmarks:
            has_detection = True
            landmarks_arr = self.landmarks_to_array(mp_result.pose_landmarks, (h, w))
        pose_result = PoseResult(
            landmarks=landmarks_arr,
            timestamp=time.time(),
            frame_width=w,
            frame_height=h,
            has_detection=has_detection,
        )
        return pose_result, mp_result

    def draw_landmarks(self, frame, landmarks):
        if landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw_styles.get_default_pose_landmarks_style(),
            )

    def landmarks_to_array(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        if not landmarks:
            return None
        arr = np.zeros((33, 3), dtype=np.float32)
        for i, lm in enumerate(landmarks.landmark):
            arr[i] = [lm.x * w, lm.y * h, lm.visibility]
        return arr

    def get_keypoint(self, landmarks, idx: int) -> Optional[Tuple[float, float, float]]:
        if landmarks is None or idx >= len(landmarks):
            return None
        return (float(landmarks[idx][0]), float(landmarks[idx][1]), float(landmarks[idx][2]))

    def close(self):
        self.pose.close()
