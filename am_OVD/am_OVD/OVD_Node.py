import numpy as np
import rclpy
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from rclpy.qos import qos_profile_sensor_data

from am_OVD.modules import DeticModule
from am_perception_utility.ImageNodeBase import ImageNodeBase

import time
import cv2


class OVDetectorNode(ImageNodeBase):
    def __init__(self, suffix="", **kwargs):
        super().__init__(**kwargs)
        self.rgb_img = None
        self.depth_img = None
        self.bridge = CvBridge()

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # self.image_publisher = self.create_publisher(Image, "/result", 1)

        self.running = False
        self.detector = None
        self.detect_result = None
        self.detect_dict = None

        self.suffix = suffix
        # cv2.namedWindow("Test"+self.suffix, cv2.WINDOW_NORMAL)

        self.timer = self.create_timer(0.05, self.timer_callback)

        # Initialization finish
        self.logger.info("Initialization finish！")
        self.last_time = time.time()

    def set_vocabulary(self, vocabulary):
        self.running = False
        del self.detector
        self.detector = DeticModule(vocabulary)
        self.running = True

    def timer_callback(self):
        if self.rgb_img is not None and self.depth_img is not None:
            if self.running:
                # Process
                pred, img = self.detector.process(self.rgb_img)
                # Timer
                current_time = time.time()
                self.logger.info("FPS: {}".format(1/(current_time-self.last_time)))
                self.last_time = current_time
                self.detect_result = img
                self.detect_dict = pred
                # img = self.bridge.cv2_to_imgmsg(self.rgb_img, "bgr8")
                # self.image_publisher.publish(img)

                # Visualization
                # cv2.imshow("Test"+self.suffix, img)

                # cv2.imwrite("result.jpg", img)

                # cv2.waitKey(50)


def main(args=None):
    rclpy.init(args=args)

    ov_detector_node = OVDetectorNode(node_name="ov_detector", rgb_enable=True, depth_enable=True)
    # custom_vocabulary = None
    custom_vocabulary = ["car"]
    ov_detector_node.set_vocabulary(custom_vocabulary)

    rclpy.spin(ov_detector_node)

    ov_detector_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
