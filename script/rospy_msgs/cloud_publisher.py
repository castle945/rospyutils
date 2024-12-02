import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
import numpy as np

class CloudPublisher:
    def __init__(self, topic_name, frame_id, queue_size=10, point_type="PointXYZ") -> None:
        self.publisher = rospy.Publisher(topic_name, PointCloud2, queue_size=queue_size)
    
        header = Header()
        header.frame_id = frame_id
        self.header = header

        assert (point_type in ["PointXYZ", "PointXYZI"])
        if point_type == "PointXYZI":
            fields = [
                        PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        PointField('intensity', 12, PointField.FLOAT32, 1),
                    ]
        self.point_type = point_type
        self.fields = fields

    def publish(self, point_cloud, stamp):
        self.header.stamp = stamp

        if self.point_type == "PointXYZ":
            cloud_msg = point_cloud2.create_cloud_xyz32(self.header, point_cloud[:, :3])
        elif self.point_type == "PointXYZI":
            cloud_msg = point_cloud2.create_cloud(self.header, self.fields, point_cloud[:, :4])

        self.publisher.publish(cloud_msg)