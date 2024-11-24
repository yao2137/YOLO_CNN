import os
import cv2
from datetime import datetime

class SnapshotSaver:
    @staticmethod
    def save_snapshot(frame, behavior, track_id):
        output_dir = "data/outputs/"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{behavior}_track_{track_id}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename