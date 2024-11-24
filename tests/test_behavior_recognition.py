import unittest
import numpy as np
from modules.behavior_recognition import BehaviorRecognition


class TestBehaviorRecognition(unittest.TestCase):
    def setUp(self):
        # Initialize the behavior recognition module
        self.recognizer = BehaviorRecognition()

        # Simulate video frames
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Simulate tracking results
        self.tracks = [
            {"track_id": 1, "bbox": [100, 200, 300, 400]},  # Initial position
            {"track_id": 2, "bbox": [400, 500, 600, 700]},  # Initial position
        ]

        # Set danger zones
        self.danger_zone = [[(100, 100), (500, 100), (500, 500), (100, 500)]]
        self.recognizer.set_danger_zones(self.danger_zone)

    def test_fall_detection(self):
        """Test fall detection"""
        # Initial state
        self.recognizer.detect_behavior(self.tracks, frame_time=0, frame=self.frame)

        # Simulate target 1 falling
        self.tracks[0]["bbox"] = [100, 300, 200, 310]  # Height drastically reduced
        behaviors = self.recognizer.detect_behavior(self.tracks, frame_time=0.5, frame=self.frame)

        # Verify results
        fall_behavior = next((b for b in behaviors if b["behavior"] == "fall"), None)
        self.assertIsNotNone(fall_behavior, "Fall detection failed.")
        self.assertEqual(fall_behavior["track_id"], 1, "Incorrect track ID for fall detection.")

    def test_static_detection(self):
        """Test static detection"""
        # Initial state
        self.recognizer.detect_behavior(self.tracks, frame_time=0, frame=self.frame)

        # Simulate target 2 being static for a threshold time
        behaviors = self.recognizer.detect_behavior(self.tracks, frame_time=6, frame=self.frame)

        # Verify results
        static_behavior = next((b for b in behaviors if b["behavior"] == "static"), None)
        self.assertIsNotNone(static_behavior, "Static detection failed.")
        self.assertEqual(static_behavior["track_id"], 2, "Incorrect track ID for static detection.")

    def test_fast_move_and_run_detection(self):
        """Test fast movement and running detection"""
        # Initial state
        self.recognizer.detect_behavior(self.tracks, frame_time=0, frame=self.frame)

        # Simulate target 1 moving fast
        self.tracks[0]["bbox"] = [800, 900, 900, 1000]  # Fast movement
        behaviors = self.recognizer.detect_behavior(self.tracks, frame_time=0.5, frame=self.frame)

        # Verify fast movement
        fast_move_behavior = next((b for b in behaviors if b["behavior"] == "fast_move"), None)
        self.assertIsNotNone(fast_move_behavior, "Fast movement detection failed.")
        self.assertEqual(fast_move_behavior["track_id"], 1, "Incorrect track ID for fast movement detection.")

        # Verify running
        run_behavior = next((b for b in behaviors if b["behavior"] == "run"), None)
        self.assertIsNotNone(run_behavior, "Run detection failed.")
        self.assertEqual(run_behavior["track_id"], 1, "Incorrect track ID for run detection.")

    def test_crawl_detection(self):
        """Test crawl detection"""
        # Initial state
        self.recognizer.detect_behavior(self.tracks, frame_time=0, frame=self.frame)

        # Simulate target 1 crawling
        self.tracks[0]["bbox"] = [100, 200, 120, 300]  # Height significantly smaller than width
        behaviors = self.recognizer.detect_behavior(self.tracks, frame_time=0.5, frame=self.frame)

        # Verify results
        crawl_behavior = next((b for b in behaviors if b["behavior"] == "crawl"), None)
        self.assertIsNotNone(crawl_behavior, "Crawl detection failed.")
        self.assertEqual(crawl_behavior["track_id"], 1, "Incorrect track ID for crawl detection.")

    def test_region_entry_exit_detection(self):
        """Test region entry/exit detection"""
        # Initial state
        self.recognizer.detect_behavior(self.tracks, frame_time=0, frame=self.frame)

        # Simulate target 1 entering the danger zone
        self.tracks[0]["bbox"] = [150, 150, 250, 250]  # Inside danger zone
        behaviors = self.recognizer.detect_behavior(self.tracks, frame_time=0.5, frame=self.frame)

        # Verify entering region
        enter_zone_behavior = next((b for b in behaviors if b["behavior"] == "enter_zone"), None)
        self.assertIsNotNone(enter_zone_behavior, "Region entry detection failed.")
        self.assertEqual(enter_zone_behavior["track_id"], 1, "Incorrect track ID for region entry detection.")

        # Simulate target 1 leaving the danger zone
        self.tracks[0]["bbox"] = [600, 600, 700, 700]  # Outside danger zone
        behaviors = self.recognizer.detect_behavior(self.tracks, frame_time=1, frame=self.frame)

        # Verify leaving region
        leave_zone_behavior = next((b for b in behaviors if b["behavior"] == "leave_zone"), None)
        self.assertIsNotNone(leave_zone_behavior, "Region exit detection failed.")
        self.assertEqual(leave_zone_behavior["track_id"], 1, "Incorrect track ID for region exit detection.")

    def test_behavior_logging_and_report(self):
        """Test behavior logging and report generation"""
        # Initial state
        self.recognizer.detect_behavior(self.tracks, frame_time=0, frame=self.frame)

        # Simulate behaviors
        self.tracks[0]["bbox"] = [100, 300, 200, 310]  # Falling
        self.recognizer.detect_behavior(self.tracks, frame_time=0.5, frame=self.frame)

        self.tracks[0]["bbox"] = [800, 900, 900, 1000]  # Fast movement
        self.recognizer.detect_behavior(self.tracks, frame_time=1, frame=self.frame)

        # Generate report
        report = self.recognizer.generate_behavior_report()
        self.assertIn(1, report, "Behavior report missing track ID 1.")
        self.assertGreater(len(report[1]), 0, "Behavior report for track ID 1 is empty.")
        self.assertEqual(report[1][0]["behavior"], "fall", "First behavior in report is incorrect.")

if __name__ == "__main__":
    unittest.main()