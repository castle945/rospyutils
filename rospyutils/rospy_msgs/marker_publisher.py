from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion
from builtin_interfaces.msg import Duration

class MarkerPublisher:
    def __init__(self, node, topic_name, frame_id, qos_profile=10):
        self.publisher = node.create_publisher(MarkerArray, topic_name, qos_profile=qos_profile)

        header = Header()
        header.frame_id = frame_id
        self.header = header

    def publish_boxes3d(self, corners_array, lines, stamp, timer_period_ns, colors=None):
        self.header.stamp = stamp

        marker_array = MarkerArray()
        for i, corners in enumerate(corners_array):
            color = colors[i] if colors is not None else [0, 1, 0]
            corner_points = [Point(x=corners[i, 0], y=corners[i, 1], z=corners[i, 2]) for i in range(8)]

            marker = Marker()
            marker.header = self.header
            marker.id = i
            marker.action = Marker.ADD
            marker.type = Marker.LINE_LIST
            marker.lifetime = Duration(nanosec=timer_period_ns)
            marker.scale.x = 0.05
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.5
            marker.points = []

            for line in lines:
                marker.points.append(corner_points[line[0]])
                marker.points.append(corner_points[line[1]])

            marker_array.markers.append(marker)
        
        self.publisher.publish(marker_array)

    def publish_text(self, points, texts, stamp, timer_period_ns, scale=[2, 2, 2], color=[0.0, 1.0, 1.0]):
        self.header.stamp = stamp

        marker_array = MarkerArray()
        for i, xyz in enumerate(points):
            marker = Marker()
            marker.header = self.header
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.lifetime = Duration(nanosec=timer_period_ns)
            marker.scale.x = scale[0]
            marker.scale.y = scale[1]
            marker.scale.z = scale[2]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            marker.pose.position.x = xyz[0]
            marker.pose.position.y = xyz[1]
            marker.pose.position.z = xyz[2]

            marker.text = texts[i]

            marker_array.markers.append(marker)

        self.publisher.publish(marker_array)
