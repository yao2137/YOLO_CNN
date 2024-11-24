import cv2
import logging
from contextlib import contextmanager

class VideoStream:
    def __init__(self, source, max_retries=5, retry_delay=2, resize=None, gray_scale=False):
        """
        Initializes the video stream.

        Args:
            source (str): Video source (RTSP/HTTP URL or local file path).
            max_retries (int): Maximum number of retries for connecting to the source.
            retry_delay (int): Delay (in seconds) between retries.
            resize (tuple): Optional. Resize frames to (width, height).
            gray_scale (bool): Optional. Convert frames to grayscale.
        """
        self.source = source
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.resize = resize
        self.gray_scale = gray_scale
        self.cap = None

        self.logger = logging.getLogger("VideoStream")
        self._connect()

    def _connect(self):
        """Establishes the video stream connection with retries."""
        attempts = 0
        while attempts < self.max_retries:
            self.logger.info(f"Attempting to connect to video source: {self.source} (Attempt {attempts + 1})")
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                self.logger.info("Successfully connected to video source.")
                return
            self.logger.warning(f"Failed to connect to video source. Retrying in {self.retry_delay} seconds...")
            attempts += 1
            time.sleep(self.retry_delay)
        raise ConnectionError(f"Unable to connect to video source after {self.max_retries} attempts.")

    def read_frame(self):
        """
        Reads a frame from the video stream.

        Returns:
            frame (numpy.ndarray): The processed video frame.
        """
        if not self.cap or not self.cap.isOpened():
            self.logger.error("Video stream is not opened. Attempting to reconnect...")
            self._connect()

        ret, frame = self.cap.read()
        if not ret:
            self.logger.warning("Failed to read frame. Skipping this iteration.")
            return None

        if self.gray_scale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.resize:
            frame = cv2.resize(frame, self.resize)

        return frame

    def release(self):
        """Releases the video stream."""
        if self.cap:
            self.cap.release()
            self.logger.info("Video stream released successfully.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()