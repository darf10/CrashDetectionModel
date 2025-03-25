import time
from threading import Thread
from ..utils.bytetrack import ByteTrack


class TrackingWorker(Thread):
    def __init__(self, detection_queue, tracking_queue):
        super().__init__()
        self.detection_queue = detection_queue
        self.tracking_queue = tracking_queue
        self.tracker = ByteTrack()
        self.running = False

    def run(self):
        while self.running:
            if not self.detection_queue.empty():
                detections = self.detection_queue.get()
                tracks = self.tracker.update(detections)
                if not self.tracking_queue.full():
                    self.tracking_queue.put(tracks)
            else:
                time.sleep(0.001)

    def stop(self):
        self.running = False
