"""
    Point Cloud Data (PCD) Visualizer
    Visualizes PCD data from all 4 LiDAR sensors (back, front, left, right)
"""

import open3d as o3d
import os

# ====== 🔧 Set your PCD file path here ======
PCD_FILE_PATH_BACK= "mcap_extraction/cam1_cal_data/pointclouds/back/back_000000_133047_449.pcd"
PCD_FILE_PATH_FRONT= "mcap_extraction/cam1_cal_data/pointclouds/front/front_000000_133047_457.pcd"
PCD_FILE_PATH_LEFT= "mcap_extraction/cam1_cal_data/pointclouds/left/left_000000_133047_513.pcd"
PCD_FILE_PATH_RIGHT= "mcap_extraction/cam1_cal_data/pointclouds/right/right_000000_133047_450.pcd"
# ============================================

def visualize_pcd(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"📂 Loading PCD file: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)

    if len(pcd.points) == 0:
        print("⚠️ Warning: Point cloud is empty.")
        return

    print(f"✅ Loaded point cloud with {len(pcd.points)} points")
    o3d.visualization.draw_geometries([pcd], window_name="PCD Viewer")

if __name__ == "__main__":
    visualize_pcd(PCD_FILE_PATH_BACK)
    visualize_pcd(PCD_FILE_PATH_FRONT)
    visualize_pcd(PCD_FILE_PATH_LEFT)
    visualize_pcd(PCD_FILE_PATH_RIGHT)
