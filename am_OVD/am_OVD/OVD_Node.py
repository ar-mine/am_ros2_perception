import numpy as np
import rclpy
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from tf2_ros import TransformBroadcaster
from rclpy.qos import qos_profile_sensor_data

from am_OVD.modules import DeticModule
from am_perception_utility.ImageNodeBase import ImageNodeBase

import time
import cv2


class OVDetectorNode(ImageNodeBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rgb_img = None
        self.depth_img = None
        self.bridge = CvBridge()

        self.tf_broadcaster = TransformBroadcaster(self)
        self.image_publisher = self.create_publisher(Image, "/result", 1)

        self.detector = DeticModule()
        cv2.namedWindow("Test", cv2.WINDOW_NORMAL)

        self.timer = self.create_timer(0.05, self.timer_callback)

        # Initialization finish
        self.logger.info("Initialization finish！")
        self.last_time = time.time()

    def timer_callback(self):
        if self.rgb_img is not None and self.depth_img is not None:
            # Process
            img = self.detector.process(self.rgb_img)
            # Timer
            current_time = time.time()
            self.logger.info("FPS: {}".format(1/(current_time-self.last_time)))
            self.last_time = current_time
            # img = self.bridge.cv2_to_imgmsg(self.rgb_img, "bgr8")
            # self.image_publisher.publish(img)

            # Visualization
            cv2.imshow("Test", img)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    ov_detector_node = OVDetectorNode(node_name="ov_detector", rgb_enable=True, depth_enable=True)

    rclpy.spin(ov_detector_node)

    ov_detector_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
