from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImagePublisher:
    def __init__(self, node, topic_name, frame_id, qos_profile=10):
        self.publisher = node.create_publisher(Image, topic_name, qos_profile=qos_profile)

        header = Header()
        header.frame_id = frame_id
        self.header = header

        self.bridge = CvBridge()

    def publish(self, image, stamp):
        self.header.stamp = stamp

        self.publisher.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))