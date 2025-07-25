#!/usr/bin/env python3
'''
# ROS 2 MCAP Extractor: Single-Camera Calibration Dataset Tool (Modified version of all_data_extractor.py)
# - Extracts images from a single camera topic (e.g., /arenacam1/images)
# - Optionally saves raw Bayer images for calibration (commented by default)
# - Extracts point clouds from selected LiDAR topics (front, left, right)
# - Saves images and point clouds in flat structure (no per-camera folders)
# - Suited for generating camera-LiDAR calibration datasets
'''

import os
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image, PointCloud2
import cv2
import numpy as np
import struct
from datetime import datetime

# How to use?
# python bagextractor.py [.mcap file] cam2_cal_data_0.mcap [storage_location]./mcap_extraction/

class BagDataExtractor(Node):
    def __init__(self):
        super().__init__('bag_data_extractor')
        
    def extract_all_data(self, bag_file, output_dir="extracted_data"):
        """Extract all images and point clouds from the bag file"""
        
        # Create organized output directories
        directory_name = "cam1_lidar" # or "cam4_cal_data" or "cam6_cal_data"

        base_dir = os.path.join(output_dir, directory_name)
        image_dir = os.path.join(base_dir, "images")
        raw_image_dir = os.path.join(base_dir, "images_raw")  # For raw Bayer data
        pcd_dir = os.path.join(base_dir, "pointclouds")
        
        # Create subdirectories for each lidar
        lidar_dirs = {
            '/rslidar_front_points': os.path.join(pcd_dir, "front"),
            # '/rslidar_back_points': os.path.join(pcd_dir, "back"),
            '/rslidar_left_points': os.path.join(pcd_dir, "left"),
            '/rslidar_right_points': os.path.join(pcd_dir, "right")
        }
        
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(raw_image_dir, exist_ok=True)
        for lidar_dir in lidar_dirs.values():
            os.makedirs(lidar_dir, exist_ok=True)
            
        # Setup bag reader
        storage_options = StorageOptions(uri=bag_file, storage_id='mcap')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        # Get topic information
        topic_types = reader.get_all_topics_and_types()
        topic_type_map = {topic.name: topic.type for topic in topic_types}
        
        print("Extracting data from topics:")
        for topic_name, topic_type in topic_type_map.items():
            print(f"  {topic_name}: {topic_type}")
        
        # Counters
        image_count = 0
        pcd_counts = {topic: 0 for topic in lidar_dirs.keys()}
        
        print(f"\nSaving data to: {base_dir}")
        print("Processing messages...")
        
        # Read all messages
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            
            topic_type = topic_type_map.get(topic)
            
            # Process images from camera
            topic_name = '/arenacam1/images' # or '/arenacam4/images' or '/arenacam2/images'
            if topic == topic_name and topic_type == 'sensor_msgs/msg/Image':
                try:
                    msg = deserialize_message(data, Image)
                    
                    # Convert to OpenCV format
                    cv_image = self.convert_ros_image_to_cv2(msg)
                    
                    if cv_image is not None:
                        ##  Storing at every 10th image
                        # if image_count % 10 == 0: (Used for intrinsics only!)

                        timestamp_sec = timestamp / 1e9
                        dt = datetime.fromtimestamp(timestamp_sec)
                        filename = f"image_{image_count:06d}_{dt.strftime('%H%M%S_%f')[:-3]}.png"
                        filepath = os.path.join(image_dir, filename)
                        
                        # Save processed image
                        cv2.imwrite(filepath, cv_image)
                        
                        # Also save raw Bayer data for a'/arenacam6/images' dvanced processing
                        # if msg.encoding.startswith('bayer'):
                        #     raw_filename = f"raw_{filename[:-4]}.png"  # Remove .png and add raw_
                        #     raw_filepath = os.path.join(raw_image_dir, raw_filename)
                        #     raw_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                        #     cv2.imwrite(raw_filepath, raw_image)
                        
                        image_count += 1
                        
                        if image_count % 100 == 0:
                            print(f"  Extracted {image_count} images...")
                    
                    else:
                        print(f"  Skipped image {image_count} due to conversion error")
                            
                except Exception as e:
                    print(f"Error processing image: {e}")
            
            # Process point clouds from lidars
            elif topic in lidar_dirs and topic_type == 'sensor_msgs/msg/PointCloud2':
                try:
                    msg = deserialize_message(data, PointCloud2)
                    
                    # Extract XYZ points
                    points = self.extract_xyz_points(msg)
                    
                    if len(points) > 0:
                        # Create filename with timestamp
                        timestamp_sec = timestamp / 1e9
                        dt = datetime.fromtimestamp(timestamp_sec)
                        lidar_name = topic.split('_')[1]  # front, back, left, right
                        filename = f"{lidar_name}_{pcd_counts[topic]:06d}_{dt.strftime('%H%M%S_%f')[:-3]}.pcd"
                        filepath = os.path.join(lidar_dirs[topic], filename)
                        
                        self.save_pcd(points, filepath)
                        pcd_counts[topic] += 1
                        
                        if sum(pcd_counts.values()) % 50 == 0:
                            print(f"  Extracted {sum(pcd_counts.values())} point clouds...")
                            
                except Exception as e:
                    print(f"Error processing point cloud from {topic}: {e}")
        
        # reader.close()
        
        print(f"\n{'='*50}")
        print("EXTRACTION COMPLETE!")
        print(f"{'='*50}")
        print(f"Images extracted: {image_count}")
        print("Point clouds extracted:")
        for topic, count in pcd_counts.items():
            lidar_name = topic.split('_')[1]
            print(f"  {lidar_name.capitalize()}: {count}")
        print(f"Total point clouds: {sum(pcd_counts.values())}")
        print(f"\nData saved to: {base_dir}")
        
        # Create a summary file
        self.create_summary_file(base_dir, image_count, pcd_counts)
        
    def convert_ros_image_to_cv2(self, msg):
        """Convert ROS Image message to OpenCV format"""
        try:
            if msg.encoding == 'rgb8':
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding == 'mono8':
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            elif msg.encoding == 'rgba8':
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
            elif msg.encoding == 'bgra8':
                cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
            # Handle Bayer formats
            elif msg.encoding == 'bayer_rggb8':
                bayer_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                cv_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_RG2BGR)
            elif msg.encoding == 'bayer_bggr8':
                bayer_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                cv_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_BG2BGR)
            elif msg.encoding == 'bayer_gbrg8':
                bayer_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                cv_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_GB2BGR)
            elif msg.encoding == 'bayer_grbg8':
                bayer_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                cv_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_GR2BGR)
            # Handle 16-bit Bayer formats
            elif msg.encoding == 'bayer_rggb16':
                bayer_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                # Convert to 8-bit for display
                bayer_image = (bayer_image / 256).astype(np.uint8)
                cv_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_RG2BGR)
            elif msg.encoding == 'bayer_bggr16':
                bayer_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                bayer_image = (bayer_image / 256).astype(np.uint8)
                cv_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_BG2BGR)
            elif msg.encoding == 'bayer_gbrg16':
                bayer_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                bayer_image = (bayer_image / 256).astype(np.uint8)
                cv_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_GB2BGR)
            elif msg.encoding == 'bayer_grbg16':
                bayer_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                bayer_image = (bayer_image / 256).astype(np.uint8)
                cv_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_GR2BGR)
            else:
                print(f"Warning: Unsupported image encoding: {msg.encoding}")
                return None
                
            return cv_image
            
        except Exception as e:
            print(f"Error converting image: {e}")
            return None
    
    def extract_xyz_points(self, pc_msg):
        """Extract XYZ points from PointCloud2 message"""
        points = []
        
        # Find field information
        x_offset = y_offset = z_offset = None
        intensity_offset = None
        point_step = pc_msg.point_step
        
        for field in pc_msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset
            elif field.name == 'intensity':
                intensity_offset = field.offset
        
        if x_offset is None or y_offset is None or z_offset is None:
            return []
        
        # Extract points
        for i in range(0, len(pc_msg.data), point_step):
            try:
                x = struct.unpack('f', pc_msg.data[i + x_offset:i + x_offset + 4])[0]
                y = struct.unpack('f', pc_msg.data[i + y_offset:i + y_offset + 4])[0]
                z = struct.unpack('f', pc_msg.data[i + z_offset:i + z_offset + 4])[0]
                
                # Get intensity if available
                intensity = 0
                if intensity_offset is not None:
                    try:
                        intensity = struct.unpack('f', pc_msg.data[i + intensity_offset:i + intensity_offset + 4])[0]
                    except:
                        intensity = 0
                
                # Filter out invalid points
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    points.append([x, y, z, intensity])
                    
            except Exception as e:
                continue
        
        return np.array(points)
    
    def save_pcd(self, points, filepath):
        """Save points as PCD file"""
        try:
            with open(filepath, 'w') as f:
                # PCD header
                f.write("# .PCD v0.7 - Point Cloud Data file format\n")
                f.write("VERSION 0.7\n")
                
                if points.shape[1] == 4:  # XYZ + intensity
                    f.write("FIELDS x y z intensity\n")
                    f.write("SIZE 4 4 4 4\n")
                    f.write("TYPE F F F F\n")
                    f.write("COUNT 1 1 1 1\n")
                else:  # XYZ only
                    f.write("FIELDS x y z\n")
                    f.write("SIZE 4 4 4\n")
                    f.write("TYPE F F F\n")
                    f.write("COUNT 1 1 1\n")
                
                f.write(f"WIDTH {len(points)}\n")
                f.write("HEIGHT 1\n")
                f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(f"POINTS {len(points)}\n")
                f.write("DATA ascii\n")
                
                # Point data
                for point in points:
                    if len(point) == 4:
                        f.write(f"{point[0]} {point[1]} {point[2]} {point[3]}\n")
                    else:
                        f.write(f"{point[0]} {point[1]} {point[2]}\n")
        except Exception as e:
            print(f"Error saving PCD file {filepath}: {e}")
    
    def create_summary_file(self, base_dir, image_count, pcd_counts):
        """Create a summary file with extraction details"""
        summary_file = os.path.join(base_dir, "extraction_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("CAM2 Calibration Data Extraction Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Extracted Data:\n")
            f.write(f"  Images: {image_count} files\n")
            f.write(f"    Location: images/\n")
            f.write(f"    Format: PNG\n\n")
            
            f.write("  Point Clouds:\n")
            for topic, count in pcd_counts.items():
                lidar_name = topic.split('_')[1]
                f.write(f"    {lidar_name.capitalize()}: {count} files\n")
            f.write(f"    Total: {sum(pcd_counts.values())} files\n")
            f.write(f"    Location: pointclouds/[front|back|left|right]/\n")
            f.write(f"    Format: PCD (ASCII)\n\n")
            
            f.write("Directory Structure:\n")
            f.write("  extracted_data/cam2_cal_data/\n")
            f.write("  ├── images/\n")
            f.write("  │   └── image_XXXXXX_HHMMSS_mmm.png\n")
            f.write("  ├── pointclouds/\n")
            f.write("  │   ├── front/\n")
            f.write("  │   ├── back/\n")
            f.write("  │   ├── left/\n")
            f.write("  │   └── right/\n")
            f.write("  └── extraction_summary.txt\n")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ros2_bag_extractor.py <bag_file.mcap> [output_dir]")
        print("Example: python ros2_bag_extractor.py cam2_cal_data_0.mcap ./my_extracted_data")
        sys.exit(1)
    
    rclpy.init()
    
    bag_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "extracted_data"
    
    if not os.path.exists(bag_file):
        print(f"Error: Bag file '{bag_file}' not found")
        sys.exit(1)
    
    print(f"Extracting data from: {bag_file}")
    print(f"Output directory: {output_dir}")
    
    extractor = BagDataExtractor()
    extractor.extract_all_data(bag_file, output_dir)
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()