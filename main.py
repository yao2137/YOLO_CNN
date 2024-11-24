import cv2
import time
import yaml
from modules.video_stream import VideoStream
from modules.object_detection import ObjectDetection
from modules.object_tracking import ObjectTracking
from modules.behavior_recognition import BehaviorRecognition
from modules.notification import Notification
from utils.config_loader import ConfigLoader
from utils.model_loader import ModelLoader
import logging


def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logger():
    """Sets up the global logger."""
    logger = logging.getLogger("Main")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def main():
    # Load configuration
    logger = setup_logger()
    config = ConfigLoader.load_config("config/config.yaml")

    yolov5_config = config["yolov5"]
    yolov5_model, confidence_threshold = ModelLoader.load_yolov5(yolov5_config)

    osnet_config = config["osnet"]
    osnet_model = ModelLoader.load_osnet(osnet_config)

    print("YOLOv5 Model:", yolov5_model)
    print("YOLOv5 Confidence Threshold:", confidence_threshold)
    print("OSNet Model:", osnet_model)

    try:
        # Initialize modules
        logger.info("Initializing modules...")
        video_stream = VideoStream(config["video_stream"]["rtsp_url"])
        detector = ObjectDetection(config["yolov5"]["model"], config["yolov5"]["confidence_threshold"])
        tracker = ObjectTracking()
        recognizer = BehaviorRecognition(
            fall_threshold=config["behavior"]["fall_threshold"],
            static_threshold=config["behavior"]["static_threshold"],
            speed_threshold=config["behavior"]["speed_threshold"]
        )
        notifier = Notification(config["notifications"])

        # Set restricted areas (if configured)
        if "danger_zones" in config["behavior"]:
            recognizer.set_danger_zones(config["behavior"]["danger_zones"])

        # Main loop
        logger.info("Starting main loop...")
        while True:
            frame_start_time = time.time()

            # Step 1: Capture frame from video stream
            frame = video_stream.read_frame()
            if frame is None:
                logger.warning("Frame capture failed, skipping this iteration.")
                continue

            # Step 2: Perform object detection
            detections = detector.detect_objects(frame)

            # Step 3: Perform object tracking
            tracks = tracker.track_objects(detections, frame)

            # Step 4: Recognize behaviors
            frame_time = time.time()
            behaviors = recognizer.detect_behavior(tracks, frame_time, frame)

            # Step 5: Notify for critical behaviors
            for behavior in behaviors:
                behavior_type = behavior["behavior"]
                track_id = behavior["track_id"]
                additional_info = behavior.get("details", {})
                notifier.notify(behavior_type, track_id, additional_info)

            # Step 6: Visualize results (optional)
            recognizer.visualize_trajectories(frame, tracks)
            cv2.imshow("Monitoring", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exiting main loop...")
                break

            # Step 7: Calculate and log frame processing time
            frame_end_time = time.time()
            frame_processing_time = frame_end_time - frame_start_time
            logger.info(f"Frame processed in {frame_processing_time:.2f} seconds.")

            # Initialize the behavior recognition system
        behavior_recognizer = BehaviorRecognition(
            detection_model_path="models/yolov5/yolov5s.pt",  # Path to YOLOv5 model
            classifier_model_path="models/behavior/behavior_classifier.pt",  # Path to ResNet18 classifier
            class_names=["eating", "sitting", "lying"],  # Behavior class names
            confidence_threshold=0.5  # Confidence threshold for YOLOv5
        )

        # Open video stream or video file
        cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with a video file path
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recognize behaviors in the frame
            behaviors = behavior_recognizer.recognize_behavior(frame)

            # Visualize the results
            for behavior in behaviors:
                x1, y1, x2, y2 = map(int, behavior["bbox"])
                label = f"{behavior['behavior']} ({behavior['confidence']:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show the frame
            cv2.imshow("Behavior Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        # Release resources
        video_stream.release()
        cv2.destroyAllWindows()
        logger.info("System shut down successfully.")


if __name__ == "__main__":
    main()