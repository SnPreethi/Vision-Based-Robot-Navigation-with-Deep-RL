#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        qos = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            history = HistoryPolicy.KEEP_LAST,
            depth = 10
        )

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            qos
        )

        self.bridge = CvBridge()

    def listener_callback(self, msg):
        self.get_logger().info('Received image data')

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image", 600, 500)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
        
        except Exception as e:
            self.get_logger().error('Could not convert ROS2 image to cv2: %s' %str(e))

        
def main(args=None):
    rclpy.init(args=args)

    img_subscriber = ImageSubscriber()

    rclpy.spin(img_subscriber)

    img_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()