import cv2


class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def get_frames(self, step_frames=30):
        """Геренатор кадрів з відео з певним кроком"""
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_idx % step_frames == 0:
                yield frame_idx, frame
            frame_idx += 1
        self.cap.release()
