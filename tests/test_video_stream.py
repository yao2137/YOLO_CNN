import unittest
from modules.video_stream import VideoStream

class TestVideoStream(unittest.TestCase):
    def setUp(self):
        self.rtsp_url = "rtsp://<camera_ip_address>:8554/unicast"  # Replace with a valid RTSP stream
        self.local_video = "test_video.mp4"  # Replace with a valid local video file

    def test_rtsp_stream(self):
        stream = VideoStream(source=self.rtsp_url)
        frame = stream.read_frame()
        self.assertIsNotNone(frame, "Failed to read frame from RTSP stream")
        stream.release()

    def test_local_video(self):
        stream = VideoStream(source=self.local_video)
        frame = stream.read_frame()
        self.assertIsNotNone(frame, "Failed to read frame from local video")
        stream.release()

    def test_frame_resize(self):
        stream = VideoStream(source=self.local_video, resize=(640, 480))
        frame = stream.read_frame()
        self.assertEqual(frame.shape[:2], (480, 640), "Frame resizing failed")
        stream.release()

    def test_grayscale_conversion(self):
        stream = VideoStream(source=self.local_video, gray_scale=True)
        frame = stream.read_frame()
        self.assertEqual(len(frame.shape), 2, "Grayscale conversion failed")
        stream.release()

if __name__ == "__main__":
    unittest.main()