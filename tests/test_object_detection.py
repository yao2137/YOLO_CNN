import unittest
import cv2
from modules.object_detection import ObjectDetection

class TestObjectDetection(unittest.TestCase):
    def setUp(self):
        self.detector = ObjectDetection(model_name="yolov5s", confidence_threshold=0.5)
        self.test_image = cv2.imread("test_image.jpg")  # Replace with a valid image path

    def test_model_loading(self):
        """Test if the YOLOv5 model loads correctly."""
        self.assertIsNotNone(self.detector.model, "YOLOv5 model failed to load.")

    def test_object_detection(self):
        """Test object detection on a sample image."""
        detections = self.detector.detect_objects(self.test_image)
        self.assertIsInstance(detections, list, "Detection output is not a list.")
        if detections:
            self.assertIn("bbox", detections[0], "Bounding box is missing in detection result.")
            self.assertIn("confidence", detections[0], "Confidence score is missing in detection result.")
            self.assertIn("class_name", detections[0], "Class name is missing in detection result.")

    def test_invalid_image(self):
        """Test detection with an invalid image input."""
        detections = self.detector.detect_objects(None)
        self.assertEqual(len(detections), 0, "Detection should return an empty list for invalid input.")

if __name__ == "__main__":
    unittest.main()