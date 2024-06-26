import numpy as np
import rclpy
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from tf2_ros import TransformBroadcaster
from rclpy.qos import qos_profile_sensor_data

from am_template_detection.DTOIDModule import DTOIDModule
from am_perception_utility.ImageNodeBase import ImageNodeBase

import time
import cv2


class ObjDetectorNode(ImageNodeBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
            mode: 0 for not processing image
                  1 for detecting with object detection
                  2 for detecting box's lattice
        """
        self.step = 1

        self.rgb_img = None
        self.depth_img = None

        self.bridge = CvBridge()
        self.image_publisher = self.create_publisher(Image, "/template_detector/test_out", 1)
        self.flag_publisher = self.create_publisher(Int32, "/template_detector/flag", 1)
        self.step_subscription = self.create_subscription(Int32, "/template_detector/step_", self.step_callback, 3)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.detector_model = DTOIDModule(template_dir="/home/armine/Codes/IMG_Automator/dataset/output")
        # self.box_detector = BoxDetector()
        self.last_time = time.time()

    def timer_callback(self):
        if self.rgb_img is not None and self.depth_img is not None:
            img = self.rgb_img.copy()

            flag = Int32()
            flag.data = 0
            angel = 0
            x_y_z = np.zeros((3,))
            if self.step == 1:
                success, img, bbox, angel, score = self.detector_model.process(self.rgb_img, threshold=0.65)
                if success:
                    depth_array = self.depth_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    # avg_depth = np.average(depth_array)/1000.0
                    # print(avg_depth)
                    avg_depth = 0.36

                    bbox_center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
                    u_v_1 = np.array([bbox_center[0], bbox_center[1], 1]).T
                    x_y_z = np.matmul(np.linalg.inv(self.camera_k), u_v_1) * avg_depth
                    # self.tf_handler(x_y_z, angel=angel)
                    self.logger.info("Current score: {}".format(score))
                    flag.data = 1
            self.tf_handler(x_y_z, angel=angel)
            self.flag_publisher.publish(flag)

            # x_y_z_list = [np.zeros((3,)), np.zeros((3,)), np.zeros((3,)), np.zeros((3,))]
            # if self.step == 2:
            #     img, bbox_list = self.box_detector.process(self.rgb_img)
            #     if len(bbox_list) == 4:
            #         for i, bbox in enumerate(bbox_list):
            #             depth_array = self.depth_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            #             avg_depth = np.average(depth_array) / 1000.0
            #
            #             bbox_center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
            #             u_v_1 = np.array([bbox_center[0], bbox_center[1], 1]).T
            #             x_y_z_list[i] = np.matmul(np.linalg.inv(self.camera_k), u_v_1) * avg_depth
            # for i, bbox in enumerate(x_y_z_list):
            #     self.tf_handler(x_y_z_list[i], child_frame="lattice_" + str(i), angel=180)

            current_time = time.time()
            cv2.putText(img, 'FPS:%.2f' % (1/(current_time - self.last_time)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(0, 0, 255), thickness=1)
            self.last_time = current_time
            # self.logger.info("Last time: {}".format(self.last_time))
            # self.logger.info("Current time: {}".format(time.time()))
            img = self.bridge.cv2_to_imgmsg(img, "bgr8")
            self.image_publisher.publish(img)

    def tf_handler(self, x_y_z, frame_id="camera_color_optical_frame", child_frame="hand_pose", angel=0):
        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame

        # Turtle only exists in 2D, thus we get x and y translation
        # coordinates from the message and set the z coordinate to 0
        t.transform.translation.x = x_y_z[0]
        t.transform.translation.y = x_y_z[1]
        t.transform.translation.z = x_y_z[2]

        q = get_quaternion_from_euler(0, 0, angel)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

    def step_callback(self, msg):
        self.step = msg.data


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


def main(args=None):
    rclpy.init(args=args)

    hand_tracker_node = ObjDetectorNode(node_name="obj_detector", rgb_enable=True, depth_enable=True)

    rclpy.spin(hand_tracker_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    hand_tracker_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
