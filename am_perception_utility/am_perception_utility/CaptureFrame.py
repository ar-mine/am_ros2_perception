import os
import cv2
from PIL import Image
import rclpy
from cv_bridge import CvBridge, CvBridgeError

from am_perception_utility.ImageNodeBase import ImageNodeBase


class CaptureFrame(ImageNodeBase):
    def __init__(self, save_path, frames=1, **kwargs):
        super().__init__(**kwargs)
        self.rgb_img = None
        self.depth_img = None
        self.bridge = CvBridge()

        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.max_frames = self.frames = frames

        self.timer = self.create_timer(0.05, self.timer_callback)
        # Initialization finish
        self.logger.info("Initialization finish！")

    def timer_callback(self):
        if self.rgb_img is not None and self.depth_img is not None:
            # Save
            rgb_img = Image.fromarray(cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2RGB))
            depth_img = Image.fromarray(self.depth_img)
            rgb_img.save(os.path.join(self.save_path, f"color_{self.max_frames-self.frames}.png"))
            depth_img.save(os.path.join(self.save_path, f"depth_{self.max_frames-self.frames}.png"))
            self.frames -= 1
            self.logger.info(f"Rest frame: {self.frames}")
            if self.frames == 0:
                self.logger.info(f"End capture!")


def main(args=None):
    rclpy.init(args=args)

    ov_detector_node = CaptureFrame(save_path="/home/armine/Pictures/RGBD",
                                    node_name="ov_detector", rgb_enable=True, depth_enable=True)

    while ov_detector_node.frames > 0:
        rclpy.spin_once(ov_detector_node)

    ov_detector_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
