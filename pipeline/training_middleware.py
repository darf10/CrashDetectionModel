import numpy as np
from threading import Thread, Event
from queue import Queue
from .pipeline import AccidentDetectionPipeline
from ..model.Model_Trainer import ModelTrainer
from ..model.Model_Config import ModelConfig


class PipelineTrainingMiddleware:
    def __init__(self, config=None):
        """
        Initialize training middleware for accident detection pipeline

        Args:
            config: Optional model configuration
        """
        self.config = config or ModelConfig()
        self.training_data = {
            'features': [],
            'labels': []
        }
        self.stop_event = Event()
        self.feature_queue = Queue()

    def _feature_extraction_worker(self, pipeline, label):
        """
        Worker to extract and collect features from pipeline

        Args:
            pipeline: Configured AccidentDetectionPipeline
            label: Binary label for the video (0 or 1)
        """
        try:
            while not self.stop_event.is_set():
                if not pipeline.result_queue.empty():
                    result = pipeline.result_queue.get()

                    # Extract meaningful features from tracks
                    frame_features = []
                    for track in result['tracks']:
                        bbox = track['bbox']
                        features = [
                            (bbox[0] + bbox[2]) / 2,  # center_x
                            (bbox[1] + bbox[3]) / 2,  # center_y
                            bbox[2] - bbox[0],  # width
                            bbox[3] - bbox[1],  # height
                            track.get('score', 1.0),  # detection confidence (with default)
                            track.get('class', -1)  # class ID (with default)
                        ]
                        frame_features.append(features)

                    # Add features to queue if not empty
                    if frame_features:
                        self.feature_queue.put({
                            'features': np.array(frame_features),
                            'label': label
                        })

                # Prevent busy waiting
                if pipeline.result_queue.empty():
                    self.stop_event.wait(0.01)
        except Exception as e:
            print(f"Feature extraction error: {e}")

    def train_from_videos(self, video_paths, labels):
        """
        Train model using video pipeline

        Args:
            video_paths: List of video file paths
            labels: Corresponding binary labels
        """
        # Reset training data
        self.training_data = {'features': [], 'labels': []}
        self.stop_event.clear()

        # Prepare feature collection threads
        extraction_threads = []
        for video_path, label in zip(video_paths, labels):
            # Create pipeline for each video
            pipeline = AccidentDetectionPipeline(video_path)

            # Start pipeline
            pipeline.start()

            # Create extraction thread
            thread = Thread(target=self._feature_extraction_worker,
                            args=(pipeline, label))
            thread.start()
            extraction_threads.append((thread, pipeline))

        # Collect features
        try:
            while len(extraction_threads) > 0:
                for thread, pipeline in list(extraction_threads):
                    # Collect features from queue
                    while not self.feature_queue.empty():
                        data = self.feature_queue.get()
                        self.training_data['features'].append(data['features'])
                        self.training_data['labels'].append(data['label'])

                    # Check if pipeline is done
                    if not pipeline.running:
                        thread.join()
                        extraction_threads.remove((thread, pipeline))
                        pipeline.stop()
        except KeyboardInterrupt:
            self.stop_event.set()
        finally:
            # Stop all pipelines
            for _, pipeline in extraction_threads:
                pipeline.stop()

        # Prepare data for training
        if self.training_data['features']:
            X = np.array(self.training_data['features'])
            y = np.array(self.training_data['labels'])

            # Train model
            trainer = ModelTrainer(self.config)
            training_history = trainer.train(X, y)

            return training_history
        else:
            print("No training data collected!")
            return None
