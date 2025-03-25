import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    """Represents a tracked object"""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    class_id: int
    age: int = 0
    lost_count: int = 0
    raw_detections: List[Dict[str, Any]] = field(default_factory=list)

    def update(self, detection):
        """Update track with new detection"""
        self.bbox = detection['bbox']
        self.score = detection['score']
        self.age += 1
        self.lost_count = 0
        self.raw_detections.append(detection)


class ByteTrack:
    def __init__(self,
                 track_buffer=30,
                 track_id_start=0,
                 match_threshold=0.8,
                 new_track_threshold=0.6):
        """
        Initialize ByteTrack tracker

        Args:
            track_buffer: Maximum number of frames a track can be lost before deletion
            track_id_start: Starting ID for tracks
            match_threshold: IoU threshold for matching tracks
            new_track_threshold: Confidence threshold for creating new tracks
        """
        self.tracks: List[Track] = []
        self._next_track_id = track_id_start
        self.track_buffer = track_buffer
        self.match_threshold = match_threshold
        self.new_track_threshold = new_track_threshold

    def update(self, detections, frame=None):
        """
        Update tracks based on new detections

        Args:
            detections: List of detections from YOLOv11m
                        Each detection is a dict with keys:
                        {'bbox': [x1,y1,x2,y2], 'score': float, 'class': int}
            frame: Optional frame (not used in this implementation,
                   but kept for compatibility)

        Returns:
            List of active tracks with their current state
        """
        # Sort detections by confidence score (high to low)
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)

        # Separate detections into high and low confidence
        high_conf_dets = [d for d in detections if d['score'] >= self.new_track_threshold]
        low_conf_dets = [d for d in detections if d['score'] < self.new_track_threshold]

        # 1. Match existing tracks with high confidence detections
        matched_tracks, unmatched_tracks, unmatched_detections = self._match_tracks(high_conf_dets)

        # 2. Update matched tracks
        for track, detection in matched_tracks:
            track.update(detection)

        # 3. Update unmatched tracks (predict and increment lost count)
        for track in unmatched_tracks:
            track.lost_count += 1
            track.age += 1

        # 4. Remove lost tracks
        self.tracks = [t for t in self.tracks if t.lost_count <= self.track_buffer]

        # 5. Initialize new tracks for unmatched high confidence detections
        for detection in unmatched_detections:
            new_track = Track(
                track_id=self._next_track_id,
                bbox=detection['bbox'],
                score=detection['score'],
                class_id=detection['class']
            )
            self.tracks.append(new_track)
            self._next_track_id += 1

        # 6. Handle low confidence detections (optional association)
        if low_conf_dets:
            self._associate_low_conf_dets(low_conf_dets)

        # Return active tracks in a standardized format
        return [
            {
                'id': track.track_id,
                'bbox': track.bbox.tolist() if isinstance(track.bbox, np.ndarray) else track.bbox,
                'score': track.score,
                'class': track.class_id
            } for track in self.tracks
        ]

    def _match_tracks(self, detections):
        """
        Match existing tracks with new detections

        Returns:
            - matched_tracks: List of (track, detection) pairs
            - unmatched_tracks: Tracks without matches
            - unmatched_detections: Detections without matches
        """
        if not self.tracks or not detections:
            return [], self.tracks, detections

        # Prepare cost matrix based on IoU
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1 - self._calculate_iou(track.bbox, det['bbox'])

        # Use Hungarian algorithm for optimal matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = []
        unmatched_tracks = list(self.tracks)
        unmatched_detections = list(detections)

        # Process matches
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < (1 - self.match_threshold):
                track = self.tracks[r]
                detection = detections[c]

                matched_tracks.append((track, detection))

                # Remove from unmatched lists
                if track in unmatched_tracks:
                    unmatched_tracks.remove(track)
                if detection in unmatched_detections:
                    unmatched_detections.remove(detection)

        return matched_tracks, unmatched_tracks, unmatched_detections

    def _associate_low_conf_dets(self, low_conf_dets):
        """
        Optional method to associate low confidence detections with existing tracks
        """
        for detection in low_conf_dets:
            best_track = None
            best_iou = self.match_threshold

            for track in self.tracks:
                iou = self._calculate_iou(track.bbox, detection['bbox'])
                if iou > best_iou:
                    best_track = track
                    best_iou = iou

            if best_track:
                # Soft update for low confidence detections
                best_track.raw_detections.append(detection)

    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union between two bounding boxes

        Args:
            bbox1, bbox2: [x1, y1, x2, y2] format
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area

        return intersection_area / union_area