import torch
import logging

from utils.model_loader import ModelLoader


class ObjectDetection:
    def __init__(self, config, use_gpu=True):
        """
        Initializes the object detection module.
        """
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

        self.logger = logging.getLogger("ObjectDetection")
        self.model, self.confidence_threshold = ModelLoader.load_yolov5(config)

    def _load_model(self):
        """Loads the YOLOv5 model using PyTorch Hub."""
        try:
            self.logger.info(f"Loading YOLOv5 model: {self.model_name} on {self.device}")
            model = torch.hub.load("ultralytics/yolov5", self.model_name).to(self.device)
            self.logger.info("YOLOv5 model loaded successfully.")
            return model
        except Exception as e:
            self.logger.error(f"Error loading YOLOv5 model: {e}")
            raise RuntimeError(f"Failed to load YOLOv5 model: {e}")

    def detect_objects(self, frame):
        """
        Performs object detection on a given frame.

        Args:
            frame (numpy.ndarray): Input image frame (BGR format).

        Returns:
            List[Dict]: List of detections with keys:
                - "bbox": [x1, y1, x2, y2] (bounding box coordinates)
                - "confidence": Detection confidence score
                - "class_id": Class ID of the detected object
                - "class_name": Class name of the detected object
        """
        try:
            # Convert frame to PyTorch tensor
            results = self.model(frame)
            detections = results.xyxy[0].cpu().numpy()
            return [
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                    "class_id": int(cls),
                    "class_name": self.model.names[int(cls)]
                }
                for x1, y1, x2, y2, conf, cls in detections if conf > self.confidence_threshold
            ]
        except Exception as e:
            self.logger.error(f"Error during object detection: {e}")
            return []

    def __del__(self):
        """Ensures resources are released when the object is deleted."""
        self.logger.info("ObjectDetection instance is being deleted.")