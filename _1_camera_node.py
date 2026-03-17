# ROS2 CAMERA SUBSCRIPTION

# importing libraries
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading

# Camera Node subscribing to camera data
class CameraNode(Node):
    def __init__(self):
        # initializations
        super().__init__('camera_node')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.lock = threading.Lock()
        self.frame_id = 0

        qos = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT,
            history = HistoryPolicy.KEEP_LAST,
            depth = 10
        )

        # Creating a subscription to the ROS topic
        self.create_subscription(Image, '/camera/image_raw', self.camera_callback, qos)

    # Camera callback method - runs everytime a new Image message is received
    def camera_callback(self, msg):
        # thread safety race condition
        with self.lock:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.frame_id += 1