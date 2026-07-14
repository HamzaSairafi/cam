from pathlib import Path

import cv2
import numpy as np


class ImageManager:
    def __init__(self, folder=None):
        self.folder = (
            Path(folder) if folder else Path(__file__).resolve().parent / "images"
        )
        # Map detected gestures to their target filenames
        self.gesture_mapping = {
            "Thumb Up": "thumb.png",
            "Peace Sign": "peace.png",
            "Fist": "fist.png",
            "Open Palm": "palm.png"
        }
        self.ensure_images_exist()

    def ensure_images_exist(self):
        """Creates the target directory and placeholders if they do not exist."""
        self.folder.mkdir(parents=True, exist_ok=True)

        for gesture, filename in self.gesture_mapping.items():
            path = self.folder / filename
            if not path.exists():
                self.create_placeholder(path, gesture)

    def create_placeholder(self, path, text):
        """Generates a simple fallback image using OpenCV text drawing."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create a dark gray 400x400 background
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img[:] = (45, 45, 45)

        # Draw the gesture name onto the placeholder frame
        cv2.putText(
            img,
            text,
            (50, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Placeholder",
            (130, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        if not cv2.imwrite(str(path), img):
            raise OSError(f"Failed to write placeholder image: {path}")

    def load_image(self, gesture_name):
        """Loads and returns the frame for a specific gesture name."""
        filename = self.gesture_mapping.get(gesture_name)
        if not filename:
            return None

        path = self.folder / filename
        if path.exists():
            image = cv2.imread(str(path))
            if image is not None:
                return image

            self.create_placeholder(path, gesture_name)
            return cv2.imread(str(path))
        return None
