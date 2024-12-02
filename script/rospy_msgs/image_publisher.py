import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImagePublisher:
    def __init__(self, topic_name, frame_id, queue_size=10) -> None:
        self.publisher = rospy.Publisher(topic_name, Image, queue_size=queue_size)

        header = Header()
        header.frame_id = frame_id
        self.header = header

        self.bridge = CvBridge()

    def publish(self, image, stamp):
        self.header.stamp = stamp

        self.publisher.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))