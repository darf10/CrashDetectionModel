import time
from threading import Thread
from ultralytics import YOLO


class DetectionWorker(Thread):
    def __init__(self, frame_queue, detection_queue, device='cuda'):
        super().__init__()
        self.frame_queue = frame_queue
        self.detection_queue = detection_queue
        self.detector = YOLO('./yolo11m.pth', device=device)
        self.running = False

    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                detections = self.detector.detect(frame)
                if not self.detection_queue.full():
                    self.detection_queue.put(detections)
            else:
                time.sleep(0.001)

    def stop(self):
        self.running = False
