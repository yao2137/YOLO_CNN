from deep_sort_realtime.deepsort_tracker import DeepSort
import logging

class ObjectTracking:
    def __init__(self, max_age=30, n_init=3, nn_budget=100):
        """
        Initializes the object tracking module.

        Args:
            max_age (int): Maximum number of frames to keep a track alive without updates.
            n_init (int): Number of consecutive detections before confirming a track.
            nn_budget (int): Maximum size of the appearance feature buffer.
        """
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget

        # Logger setup
        self.logger = logging.getLogger("ObjectTracking")
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Initialize Deep SORT tracker
        self.tracker = self._initialize_tracker()

    def _initialize_tracker(self):
        """Initializes the Deep SORT tracker."""
        self.logger.info("Initializing Deep SORT tracker...")
        try:
            tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                nn_budget=self.nn_budget
            )
            self.logger.info("Deep SORT tracker initialized successfully.")
            return tracker
        except Exception as e:
            self.logger.error(f"Error initializing Deep SORT tracker: {e}")
            raise RuntimeError(f"Failed to initialize tracker: {e}")

    def track_objects(self, detections, frame):
        """
        Updates the tracker with new detections and returns tracking results.

        Args:
            detections (List[Dict]): List of detections with keys:
                - "bbox": [x1, y1, x2, y2]
                - "confidence": Detection confidence score.
            frame (numpy.ndarray): Current video frame (used for ReID).

        Returns:
            List[Dict]: List of tracking results with keys:
                - "track_id": Unique ID of the tracked object.
                - "bbox": [x1, y1, x2, y2] (bounding box coordinates).
        """
        try:
            # Prepare detections in Deep SORT format
            deep_sort_detections = [
                det["bbox"] + [det["confidence"]] for det in detections
            ]

            # Update tracker
            tracks = self.tracker.update_tracks(deep_sort_detections, frame=frame)

            # Format tracking results
            tracked_objects = []
            for track in tracks:
                if track.is_confirmed() and track.time_since_update <= 1:
                    tracked_objects.append({
                        "track_id": track.track_id,
                        "bbox": track.to_tlbr()  # Convert to [top, left, bottom, right]
                    })

            self.logger.info(f"Tracking results: {tracked_objects}")
            return tracked_objects
        except Exception as e:
            self.logger.error(f"Error during object tracking: {e}")
            return []

    def __del__(self):
        """Ensure proper cleanup."""
        self.logger.info("ObjectTracking instance is being deleted.")