import cv2
import time
from threading import Thread


class VideoCaptureWorker(Thread):
    def __init__(self, video_source, frame_queue):
        super().__init__()
        self.video_source = video_source
        self.frame_queue = frame_queue
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"Error: Cannot open video {self.video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_time = 1.0 / fps

        while self.running:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

            elapsed = time.time() - start_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed) # Need to change to supervisor implementation

        cap.release()

    def stop(self):
        self.running = False
