from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

class CloudPublisher:
    def __init__(self, node, topic_name, frame_id, qos_profile=10, point_type="PointXYZ"):
        self.publisher = node.create_publisher(PointCloud2, topic_name, qos_profile=qos_profile)
    
        header = Header()
        header.frame_id = frame_id
        self.header = header

        assert (point_type in ["PointXYZ", "PointXYZI"])
        if point_type == "PointXYZI":
            fields = [
                        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
                    ]
            self.fields = fields
        self.point_type = point_type

    def publish(self, point_cloud, stamp):
        self.header.stamp = stamp

        if self.point_type == "PointXYZ":
            cloud_msg = point_cloud2.create_cloud_xyz32(self.header, point_cloud[:, :3])
        elif self.point_type == "PointXYZI":
            cloud_msg = point_cloud2.create_cloud(self.header, self.fields, point_cloud[:, :4])

        self.publisher.publish(cloud_msg)