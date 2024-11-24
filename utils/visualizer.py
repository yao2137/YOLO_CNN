import cv2

class Visualizer:
    @staticmethod
    def draw_bounding_box(frame, bbox, label, color=(0, 255, 0)):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    @staticmethod
    def draw_trajectory(frame, points, color=(255, 0, 0)):
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], color, 2)