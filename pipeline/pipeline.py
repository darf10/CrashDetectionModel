from workers.capture import VideoCaptureWorker
from workers.tracking import TrackingWorker
from workers.analysis import AnalysisWorker
from queue import Queue
from threading import Lock


class AccidentDetectionPipeline:
    def __init__(self, video_source, gpu_id=0, buffer_size=30):
        self.video_source = video_source
        self.frame_queue = Queue(maxsize=5)
        self.tracking_queue = Queue(maxsize=5)
        self.result_queue = Queue()
        self.motion_history = {}
        self.buffer_size = buffer_size
        self.motion_lock = Lock()
        self.running = False

        self.capture_worker = VideoCaptureWorker(video_source, self.frame_queue)
        self.tracking_worker = TrackingWorker(self.frame_queue, self.tracking_queue)
        self.analysis_worker = AnalysisWorker(self.tracking_queue, self.result_queue, self.motion_lock,
                                              self.motion_history, buffer_size)

    def start(self):
        self.running = True
        self.capture_worker.start()
        self.tracking_worker.start()
        self.analysis_worker.start()

    def stop(self):
        self.running = False
        self.capture_worker.stop()
        self.tracking_worker.stop()
        self.analysis_worker.stop()
