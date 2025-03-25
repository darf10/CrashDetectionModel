import time
from threading import Thread
from ultralytics import YOLO


class TrackingWorker(Thread):
    def __init__(self, frame_queue, tracking_queue, device='cuda'):
        super().__init__()
        self.frame_queue = frame_queue
        self.tracking_queue = tracking_queue
        self.detector = YOLO('./yolo11m.pth', device=device)
        self.running = False

    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                # Use YOLO's track method directly on the frame
                tracks = self.detector.track(frame, persist=True)

                if tracks is not None and not self.tracking_queue.full():
                    # Convert YOLO tracks to our existing track format
                    formatted_tracks = [
                        {
                            'id': int(track[4]) if len(track) > 4 else -1,
                            'bbox': track[:4].tolist(),
                            'score': float(track[5]) if len(track) > 5 else 1.0,
                            'class': int(track[6]) if len(track) > 6 else -1
                        }
                        for track in tracks.xyxy[0].cpu().numpy()
                    ]

                    self.tracking_queue.put(formatted_tracks)
            else:
                time.sleep(0.001)

    def stop(self):
        self.running = False
