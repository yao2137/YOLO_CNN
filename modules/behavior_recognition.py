import time
import logging
import cv2
import numpy as np
from shapely.geometry import Point, Polygon  # For region checks
from torchvision import transforms, models
import torch
from PIL import Image
from utils.snapshot_saver import SnapshotSaver


class BehaviorRecognition:
    def __init__(self, behavior_model_path, class_names, fall_threshold=0.4, static_threshold=5,
                 speed_threshold=2.0, run_threshold=4.0, crawl_ratio=0.5, device=None):
        """
        Initializes the behavior recognition module.

        Args:
            behavior_model_path (str): Path to the ResNet-based behavior classification model.
            class_names (list): List of behavior class names.
            fall_threshold (float): Height reduction ratio threshold for fall detection.
            static_threshold (int): Time threshold (in seconds) for static detection.
            speed_threshold (float): Speed threshold for fast movement detection.
            run_threshold (float): Speed threshold for run detection.
            crawl_ratio (float): Height-to-width ratio threshold for crawl detection.
            device (str): Device to run the models on ('cuda' or 'cpu').
        """
        self.fall_threshold = fall_threshold
        self.static_threshold = static_threshold
        self.speed_threshold = speed_threshold
        self.run_threshold = run_threshold
        self.crawl_ratio = crawl_ratio
        self.class_names = class_names

        # Initialize device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load behavior classification model
        self.behavior_model = self._load_behavior_model(behavior_model_path, len(class_names))

        # To track target states over time
        self.target_states = {}
        self.behavior_logs = {}
        self.danger_zones = []  # List of polygons defining restricted areas

        # Logger
        self.logger = logging.getLogger("BehaviorRecognition")
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Background modeling for environment adaptation
        self.background_model = None

        # Image transformation for behavior classification
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def set_danger_zones(self, zones):
        """
        Sets the restricted/danger zones.

        Args:
            zones (List[List[Tuple]]): List of polygons where each polygon is a list of (x, y) coordinates.
        """
        self.danger_zones = [Polygon(zone) for zone in zones]
        self.logger.info("Danger zones set successfully.")

    def check_region_entry(self, target_center):
        """
        Checks if the target has entered or left a danger zone.

        Args:
            target_center (Tuple): The current center of the target.

        Returns:
            str: "enter" or "leave" if the target enters/leaves a danger zone; None otherwise.
        """
        point = Point(target_center)
        for zone in self.danger_zones:
            if zone.contains(point):
                return "enter"
        return "leave"

    def adjust_thresholds(self, frame):
        """
        Dynamically adjusts thresholds based on environmental conditions.

        Args:
            frame (numpy.ndarray): Current video frame.
        """
        if self.background_model is None:
            self.background_model = cv2.createBackgroundSubtractorMOG2()

        fg_mask = self.background_model.apply(frame)
        noise_level = np.mean(fg_mask) / 255.0  # Normalized noise level (0 to 1)
        self.logger.info(f"Environmental noise level: {noise_level:.2f}")

        # Dynamically adjust thresholds based on noise
        self.fall_threshold = 0.4 + noise_level * 0.1
        self.speed_threshold = 2.0 + noise_level * 0.5

    def _load_behavior_model(self, model_path, num_classes):
        """
        Loads the ResNet18 behavior classification model.

        Args:
            model_path (str): Path to the model file.
            num_classes (int): Number of behavior classes.

        Returns:
            torch.nn.Module: Loaded model.
        """
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def log_behavior(self, track_id, behavior, timestamp):
        """
        Logs the behavior of a target.

        Args:
            track_id (int): The ID of the tracked target.
            behavior (str): The detected behavior.
            timestamp (float): The time of detection.
        """
        if track_id not in self.behavior_logs:
            self.behavior_logs[track_id] = []
        self.behavior_logs[track_id].append({"behavior": behavior, "time": timestamp})
        self.logger.info(f"Behavior logged for track ID {track_id}: {behavior} at {timestamp:.2f}")

    def generate_behavior_report(self):
        """
        Generates a behavior report for all tracked targets.

        Returns:
            Dict: A dictionary containing behavior time lines for each target.
        """
        return self.behavior_logs

    def classify_behavior(self, frame, bbox):
        """
        Classifies behavior using the behavior classification model.

        Args:
            frame (numpy.ndarray): Current video frame.
            bbox (Tuple): Bounding box of the target (top, left, bottom, right).

        Returns:
            str: Classified behavior.
        """
        top, left, bottom, right = bbox
        cropped_img = frame[top:bottom, left:right]

        try:
            pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            # Predict behavior
            with torch.no_grad():
                output = self.behavior_model(input_tensor)
                _, predicted = torch.max(output, 1)
                return self.class_names[predicted.item()]
        except Exception as e:
            self.logger.error(f"Error in behavior classification: {e}")
            return "unknown"

    def detect_behavior(self, tracks, frame_time, frame):
        """
        Detects behaviors based on tracking data.

        Args:
            tracks (List[Dict]): Tracking results.
            frame_time (float): Timestamp of the current frame.
            frame (numpy.ndarray): Current video frame.

        Returns:
            List[Dict]: Detected behaviors with metadata.
        """
        detected_behaviors = []

        for track in tracks:
            track_id = track["track_id"]
            bbox = track["bbox"]
            top, left, bottom, right = bbox

            current_height = bottom - top
            current_width = right - left
            current_center = ((left + right) / 2, (top + bottom) / 2)

            if track_id not in self.target_states:
                self.target_states[track_id] = {
                    "last_height": current_height,
                    "last_center": current_center,
                    "last_time": frame_time,
                    "static_start": None,
                }
                continue

            prev_state = self.target_states[track_id]
            prev_height = prev_state["last_height"]
            prev_center = prev_state["last_center"]
            prev_time = prev_state["last_time"]
            time_diff = frame_time - prev_time

            # Fall Detection
            height_ratio = current_height / prev_height if prev_height > 0 else 1.0
            if height_ratio < self.fall_threshold and time_diff < 1.0:
                detected_behaviors.append({"track_id": track_id, "behavior": "fall"})
                self.log_behavior(track_id, "fall", frame_time)

            # Static Detection
            if current_center == prev_center:
                if prev_state["static_start"] is None:
                    prev_state["static_start"] = frame_time
                elif frame_time - prev_state["static_start"] > self.static_threshold:
                    detected_behaviors.append({"track_id": track_id, "behavior": "static"})
                    self.log_behavior(track_id, "static", frame_time)
            else:
                prev_state["static_start"] = None

            # Fast Move and Run Detection
            distance = ((current_center[0] - prev_center[0]) ** 2 + (current_center[1] - prev_center[1]) ** 2) ** 0.5
            speed = distance / time_diff if time_diff > 0 else 0.0
            if speed > self.speed_threshold:
                detected_behaviors.append({"track_id": track_id, "behavior": "fast_move"})
                self.log_behavior(track_id, "fast_move", frame_time)
            if speed > self.run_threshold:
                detected_behaviors.append({"track_id": track_id, "behavior": "run"})
                self.log_behavior(track_id, "run", frame_time)

            # Crawl Detection
            height_width_ratio = current_height / current_width if current_width > 0 else 1.0
            if height_width_ratio < self.crawl_ratio:
                detected_behaviors.append({"track_id": track_id, "behavior": "crawl"})
                self.log_behavior(track_id, "crawl", frame_time)

            # Region Entry/Exit Detection
            region_status = self.check_region_entry(current_center)
            if region_status == "enter":
                self.log_behavior(track_id, "enter_zone", frame_time)
            elif region_status == "leave":
                self.log_behavior(track_id, "leave_zone", frame_time)

            # Update target state
            self.target_states[track_id].update({
                "last_height": current_height,
                "last_center": current_center,
                "last_time": frame_time,
            })

        # Adjust thresholds based on environment
        self.adjust_thresholds(frame)

        for behavior in detected_behaviors:
            if behavior["behavior"] == "fall":
                SnapshotSaver.save_snapshot(frame, behavior["behavior"], behavior["track_id"])
        return detected_behaviors

    def visualize_trajectories(self, frame, tracks):
        """
        Visualizes trajectories and behaviors on the video frame.

        Args:
            frame (numpy.ndarray): The video frame.
            tracks (List[Dict]): Tracking results.
        """
        for track in tracks:
            bbox = track["bbox"]
            track_id = track["track_id"]
            top, left, bottom, right = bbox

            # Draw bounding box and ID
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
