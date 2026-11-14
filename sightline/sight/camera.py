"""Live mode's camera: grab frames straight from the laptop webcam with OpenCV.

Snapshot mode uses the browser (`st.camera_input`). Live mode can't: the
browser widget only fires when you click. Because Streamlit runs on your own
laptop, the Python process can open the camera itself, and a timed Streamlit
fragment asks it for a frame every N seconds.

macOS asks for camera permission the first time, for the app that LAUNCHED
Python (Terminal, iTerm, VS Code...), not for the browser.
"""

import cv2
from PIL import Image


class Camera:
    def __init__(self, index: int = 0):
        self.index = index
        self.cap = cv2.VideoCapture(index)   # 0 = the built-in camera

    def is_open(self) -> bool:
        return self.cap.isOpened()

    def grab(self) -> Image.Image | None:
        """One frame as an RGB PIL image, or None if the camera gave nothing."""
        # The driver keeps a small queue of frames. Between our samples it fills
        # with stale ones, so throw a few away to get what's in front of it NOW.
        for _ in range(3):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        # OpenCV hands back BGR channel order; PIL (and Jev's eyes) expect RGB.
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def release(self) -> None:
        self.cap.release()
