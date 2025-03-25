import time
import numpy as np
from threading import Thread
from ...model.Accident_Classifier import AccidentClassifier


class AnalysisWorker(Thread):
    def __init__(self, tracking_queue, result_queue, motion_lock, motion_history, buffer_size):
        super().__init__()
        self.tracking_queue = tracking_queue
        self.result_queue = result_queue
        self.motion_lock = motion_lock
        self.motion_history = motion_history
        self.accident_classifier = AccidentClassifier()
        self.buffer_size = buffer_size
        self.running = False

    def run(self):
        while self.running:
            if not self.tracking_queue.empty():
                tracks = self.tracking_queue.get()
                accident_detected, accident_info = self._detect_accidents(tracks)

                result = {
                    'tracks': tracks,
                    'accident_detected': accident_detected,
                    'accident_info': accident_info if accident_detected else None
                }
                self.result_queue.put(result)
            else:
                time.sleep(0.001)  # Need to change to supervisor implementation

    def _detect_accidents(self, tracks):
        with self.motion_lock:
            for track in tracks:
                track_id = track['id']
                bbox = track['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2

                if track_id not in self.motion_history:
                    self.motion_history[track_id] = {'positions': [], 'timestamps': []}

                self.motion_history[track_id]['positions'].append((center_x, center_y))
                self.motion_history[track_id]['timestamps'].append(time.time())

                if len(self.motion_history[track_id]['positions']) > self.buffer_size:
                    self.motion_history[track_id]['positions'].pop(0)
                    self.motion_history[track_id]['timestamps'].pop(0)

                # Example accident detection: sudden stop
                if len(self.motion_history[track_id]['positions']) > 2:
                    dx = np.linalg.norm(np.array(self.motion_history[track_id]['positions'][-1]) -
                                        np.array(self.motion_history[track_id]['positions'][-2]))
                    dt = self.motion_history[track_id]['timestamps'][-1] - self.motion_history[track_id]['timestamps'][
                        -2]

                    if dt > 0 and dx / dt < 0.5:  # Example threshold
                        return True, {'track_id': track_id, 'type': 'sudden stop'}

        return False, None

    def stop(self):
        self.running = False
