import sys
sys.path.append("/home/mcity/Downloads/mcap/python/mcap-ros2-support")

import os
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
from mcap.reader import make_reader


IMG_OUT_DIR = os.path.expanduser("~/mcap_extraction/images")
PCD_OUT_DIR = os.path.expanduser("~/mcap_extraction/pointclouds")
MCAP_PATH = "/home/mcity/Downloads/cam2_cal_data_0.mcap"  # Change this

bridge = CvBridge()

def save_image(msg: Image, timestamp_ns):
    cv_image = bridge.imgmsg_to_cv2(msg)
    filename = os.path.join(IMG_OUT_DIR, f"img_{timestamp_ns}.png")
    import cv2
    cv2.imwrite(filename, cv_image)
    print(f"✅ Saved {filename}")

def save_pointcloud(msg: PointCloud2, timestamp_ns):
    points = list(point_cloud2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True))
    if not points:
        return
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.array(points))
    filename = os.path.join(PCD_OUT_DIR, f"cloud_{timestamp_ns}.pcd")
    o3d.io.write_point_cloud(filename, pc)
    print(f"✅ Saved {filename}")

def main():
    rclpy.init()
    with open(MCAP_PATH, "rb") as f:
        reader = make_reader(f)
        for schema, channel, msg in reader.iter_decoded_messages():
            topic = channel.topic
            timestamp_ns = msg.log_time

            if topic == "/arenacam2/images":
                img_msg = deserialize_message(msg.message, get_message("sensor_msgs/msg/Image"))
                save_image(img_msg, timestamp_ns)

            elif topic in ["/velodyne_points", "/points_raw"]:
                pc_msg = deserialize_message(msg.message, get_message("sensor_msgs/msg/PointCloud2"))
                save_pointcloud(pc_msg, timestamp_ns)


    rclpy.shutdown()

if __name__ == "__main__":
    main()
