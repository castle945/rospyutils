import rospy
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion
import numpy as np

class MarkerPublisher:
    def __init__(self, topic_name, frame_id, queue_size=10):
        self.publisher = rospy.Publisher(topic_name, MarkerArray, queue_size=queue_size)

        header = Header()
        header.frame_id = frame_id
        self.header = header

    def publish_boxes3d(self, corners_array, lines, stamp, lifetime, colors=None):

        self.header.stamp = stamp

        marker_array = MarkerArray()
        for i, corners in enumerate(corners_array):
            color = colors[i] if colors is not None else [0, 1, 0]
            corner_points = [Point(corners[i, 0], corners[i, 1], corners[i, 2]) for i in range(8)]

            marker = Marker()
            marker.header = self.header
            marker.id = i
            marker.action = Marker.ADD
            marker.type = Marker.LINE_LIST
            marker.lifetime = lifetime
            marker.scale.x = 0.05
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.5
            marker.pose.orientation = Quaternion(0, 0, 0, 1) # It won't be used, just to avoid warnings
            marker.points = []

            for line in lines:
                marker.points.append(corner_points[line[0]])
                marker.points.append(corner_points[line[1]])

            marker_array.markers.append(marker)
        
        self.publisher.publish(marker_array)

    def publish_text(self, points, texts, stamp, lifetime, scale=[2, 2, 2], color=[0, 1, 1]):
        self.header.stamp = stamp

        marker_array = MarkerArray()
        for i, xyz in enumerate(points):
            marker = Marker()
            marker.header = self.header
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.lifetime = lifetime
            marker.scale.x = scale[0]
            marker.scale.y = scale[1]
            marker.scale.z = scale[2]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1
            marker.pose.position.x = xyz[0]
            marker.pose.position.y = xyz[1]
            marker.pose.position.z = xyz[2]
            marker.pose.orientation = Quaternion(0, 0, 0, 1) # It won't be used, just to avoid warnings

            marker.text = texts[i]

            marker_array.markers.append(marker)

        self.publisher.publish(marker_array)
