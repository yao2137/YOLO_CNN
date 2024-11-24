import torch
import numpy as np

class YOLOv5:
    def __init__(self, model_path, device=None, confidence_threshold=0.5):
        """
        Initialize the YOLOv5 model.

        Args:
            model_path (str): Path to the YOLOv5 model file (.pt).
            device (str): Device to run the model on ('cuda' or 'cpu'). Defaults to None.
            confidence_threshold (float): Confidence threshold for filtering detections.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, source="local").to(self.device)
        self.confidence_threshold = confidence_threshold

    def predict(self, frame):
        """
        Perform object detection on a single frame.

        Args:
            frame (numpy.ndarray): Input frame in BGR format.

        Returns:
            List[Dict]: List of detections, each containing:
                - "bbox": [x1, y1, x2, y2]
                - "confidence": Confidence score
                - "class_id": Class index
                - "class_name": Class name
        """
        # Convert the frame to a format compatible with YOLOv5
        results = self.model(frame)

        # Process detections
        detections = results.xyxy[0].cpu().numpy()  # Extract detections as NumPy array
        outputs = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            if conf >= self.confidence_threshold:
                outputs.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                    "class_id": int(class_id),
                    "class_name": self.model.names[int(class_id)]
                })

        return outputs