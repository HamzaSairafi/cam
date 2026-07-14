import cv2
import time
from gesture_detector import GestureDetector
from image_manager import ImageManager


def main():
    detector = None
    cap = None

    try:
        # Instantiate modules with error boundaries
        detector = GestureDetector()
        image_manager = ImageManager()

        # Establish webcam capture stream hook
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(
                "Error: Video capture stream unreadable. "
                "Check camera permissions/connections."
            )
            return

        prev_time = 0
        last_gesture = None
        viewer_window_open = False

        print("Application Active. Target standard output console or frame focus window...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Dropped device tracking frames.")
                break

            # Horizontal reflection transformation for natural previewing (Mirror Effect)
            frame = cv2.flip(frame, 1)

            # Process frame
            gesture, frame = detector.detect_gesture(frame)

            # Separate Viewer Lifecycle Event Controller
            if gesture != last_gesture:
                if gesture in image_manager.gesture_mapping:
                    img = image_manager.load_image(gesture)
                    if img is not None:
                        cv2.imshow("Gesture Viewer", img)
                        viewer_window_open = True
                else:
                    # Close the display viewport if gesture is unrecognized/lost
                    if viewer_window_open:
                        try:
                            cv2.destroyWindow("Gesture Viewer")
                        except cv2.error:
                            pass
                        viewer_window_open = False

                last_gesture = gesture

            # Continuous health poll to ensure manual window closes do not break the runtime cycle
            if viewer_window_open:
                try:
                    window_visible = cv2.getWindowProperty(
                        "Gesture Viewer",
                        cv2.WND_PROP_VISIBLE,
                    )
                    if window_visible < 1:
                        viewer_window_open = False
                        last_gesture = None
                except cv2.error:
                    viewer_window_open = False
                    last_gesture = None

            # Compute runtime processing frame performance rates
            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Render interface HUD directly onto output context stream
            cv2.rectangle(frame, (15, 15), (320, 115), (0, 0, 0), -1)
            cv2.rectangle(frame, (15, 15), (320, 115), (50, 205, 50), 1)

            cv2.putText(
                frame,
                f"FPS: {fps}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Press 'Q' to Exit",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

            cv2.imshow("Webcam Feed", frame)

            # Break processing loops instantly on keyboard interrupt event registrations
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        if cap is not None:
            cap.release()
        if detector is not None:
            detector.defer_close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

if __name__ == "__main__":
    main()
