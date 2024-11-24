import unittest
import numpy as np
from modules.object_tracking import ObjectTracking

class TestObjectTracking(unittest.TestCase):
    def setUp(self):
        self.tracker = ObjectTracking(max_age=10, n_init=3, nn_budget=50)
        self.test_detections = [
            {"bbox": [100, 200, 300, 400], "confidence": 0.9},
            {"bbox": [400, 500, 600, 700], "confidence": 0.8}
        ]
        self.test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_tracker_initialization(self):
        """Test if the tracker initializes correctly."""
        self.assertIsNotNone(self.tracker.tracker, "Tracker failed to initialize.")

    def test_tracking(self):
        """Test tracking results with sample detections."""
        results = self.tracker.track_objects(self.test_detections, self.test_frame)
        self.assertIsInstance(results, list, "Tracking output is not a list.")
        if results:
            self.assertIn("track_id", results[0], "Track ID is missing in tracking result.")
            self.assertIn("bbox", results[0], "Bounding box is missing in tracking result.")

    def test_empty_detections(self):
        """Test tracker with no detections."""
        results = self.tracker.track_objects([], self.test_frame)
        self.assertEqual(len(results), 0, "Tracking results should be empty for no detections.")

if __name__ == "__main__":
    unittest.main()