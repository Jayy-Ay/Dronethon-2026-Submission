import unittest
import cv2
from vision.frame_provider import WebcamFrameProvider, VideoFileFrameProvider
from vision.depth_processor import estimate_depth, frame_to_point_cloud

class TestFrameProvider(unittest.TestCase):
    def test_webcam_provider_init(self):
        provider = WebcamFrameProvider()
        self.assertIsInstance(provider, WebcamFrameProvider)
        provider.release()

    def test_video_file_provider_init(self):
        provider = VideoFileFrameProvider('test.mp4')
        self.assertIsInstance(provider, VideoFileFrameProvider)
        provider.release()

class TestDepthProcessor(unittest.TestCase):
    def test_estimate_depth_shape(self):
        # Create dummy image
        frame = cv2.imread('tests/test_image.jpg')
        if frame is None:
            frame = (255 * np.ones((240, 320, 3), dtype=np.uint8))
        depth = estimate_depth(frame)
        self.assertEqual(depth.shape[:2], frame.shape[:2])

    def test_frame_to_point_cloud(self):
        frame = (255 * np.ones((240, 320, 3), dtype=np.uint8))
        pc = frame_to_point_cloud(frame)
        self.assertTrue(hasattr(pc, 'points'))

if __name__ == '__main__':
    unittest.main()
