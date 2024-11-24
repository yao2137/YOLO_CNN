import cv2
from modules.yolov5_interface import YOLOv5

def main():
    # Initialize the YOLOv5 model
    model_path = "models/yolov5/best.pt"
    yolov5 = YOLOv5(model_path=model_path, confidence_threshold=0.5)

    # Open a video or webcam
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video file path
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        detections = yolov5.predict(frame)

        # Visualize detections
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["confidence"]
            class_name = det["class_name"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv5 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()