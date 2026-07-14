from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np


_DEFERRED_LANDMARKERS = []


class GestureDetector:
    HAND_CONNECTIONS = (
        (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (0, 17),
        (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
    )

    def __init__(self, model_path=None):
        default_model_path = (
            Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
        )
        self.model_path = Path(model_path) if model_path else default_model_path
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Missing MediaPipe hand model: {self.model_path}. "
                "Download the HandLandmarker task model to this path."
            )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect_gesture(self, frame):
        """
        Processes BGR frames, displays overlay points, and evaluates poses.
        Returns a tuple of (gesture_name, annotated_frame).
        """
        # MediaPipe requires RGB format input.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )
        results = self.landmarker.detect_for_video(mp_image, self._next_timestamp_ms())

        gesture_name = "None"

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]

            if len(hand_landmarks) >= 21:
                self._draw_landmarks(frame, hand_landmarks)
                gesture_name = self._analyze_landmarks(hand_landmarks)
            else:
                gesture_name = "Unknown"

        return gesture_name, frame

    def close(self):
        """Release the native MediaPipe task resources."""
        if getattr(self, "landmarker", None) is not None:
            self.landmarker.close()
            self.landmarker = None

    def defer_close(self):
        """Keep MediaPipe alive until process shutdown to avoid slow native close."""
        if getattr(self, "landmarker", None) is not None:
            _DEFERRED_LANDMARKERS.append(self.landmarker)
            self.landmarker = None

    def _next_timestamp_ms(self):
        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _draw_landmarks(self, frame, landmarks):
        height, width = frame.shape[:2]
        points = [
            (int(landmark.x * width), int(landmark.y * height))
            for landmark in landmarks
        ]

        for start, end in self.HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (80, 220, 80), 2)

        for point in points:
            cv2.circle(frame, point, 4, (0, 255, 255), -1)

    def _analyze_landmarks(self, landmarks):
        """
        Heuristic evaluation parsing landmark structures relative to joints.
        """
        # Track 4 main fingers: Index, Middle, Ring, Pinky
        # Logic evaluates if Tip coordinate Y is above internal PIP joint Y coordinate.
        fingers = []
        fingers.append(landmarks[8].y < landmarks[6].y)  # Index Open
        fingers.append(landmarks[12].y < landmarks[10].y)  # Middle Open
        fingers.append(landmarks[16].y < landmarks[14].y)  # Ring Open
        fingers.append(landmarks[20].y < landmarks[18].y)  # Pinky Open

        # Thumb logic check: Determine if Thumb Tip (4) sits well higher than Index MCP joint base (5)
        thumb_is_up = landmarks[4].y < landmarks[5].y

        # Match structural profiles to the target gestures
        if fingers == [True, True, True, True]:
            return "Open Palm"
        elif fingers == [True, True, False, False]:
            return "Peace Sign"
        elif fingers == [False, False, False, False]:
            if thumb_is_up:
                return "Thumb Up"
            else:
                return "Fist"

        return "Unknown"

    def __del__(self):
        try:
            self.defer_close()
        except Exception:
            pass
