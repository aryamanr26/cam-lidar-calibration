'''
python3 your_script.py \
  --folder cam_lidar_data \
  --base_path july8 \
  --img_dir july8/cam_lidar_data/images/cam6 \
  --pcd_dir july8/cam_lidar_data/pointclouds/front \
  --sync_img_dir july8/cam_lidar_data/synced/cam6/images \
  --sync_pcd_dir july8/cam_lidar_data/synced/cam6/frontpointclouds

'''
import os
import glob
import shutil
import argparse

## CHANGE ALL DIRECTORIES BEFORE RUNNING
parser = argparse.ArgumentParser(description="Synchronize camera and LiDAR data directories.")
parser.add_argument('--base_path', required=True, help='Base path (e.g., july8/cam_lidar_data)')
parser.add_argument('--img_dir', help='Image directory path')
parser.add_argument('--pcd_dir', help='PointCloud directory path')
parser.add_argument('--sync_img_dir', help='Output synced image directory')
parser.add_argument('--sync_pcd_dir', help='Output synced pointcloud directory')

args = parser.parse_args()

# Set default paths if not provided
IMG_DIR = args.base_path + args.img_dir or f"{args.base_path}/images/cam6"
PCD_DIR = args.base_path + args.pcd_dir or f"{args.base_path}/pointclouds/front"
SYNC_IMG_DIR = args.base_path + args.sync_img_dir or f"{args.base_path}synced/cam6/images"
SYNC_PCD_DIR = args.base_path + args.sync_pcd_dir or f"{args.base_path}synced/cam6/frontpointclouds"

# Print confirmation
print("\n--- Directory Configuration ---")
print(f"Image Directory: {IMG_DIR}")
print(f"Pointcloud Directory: {PCD_DIR}")
print(f"Synced Image Directory: {SYNC_IMG_DIR}")
print(f"Synced Pointcloud Directory: {SYNC_PCD_DIR}")
print("--------------------------------\n")

os.makedirs(SYNC_IMG_DIR, exist_ok=True)
os.makedirs(SYNC_PCD_DIR, exist_ok=True)

img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
pcd_files = sorted(glob.glob(os.path.join(PCD_DIR, "*.pcd"))) # Change file name

def ros_time_string_to_seconds(hour, minute, second, frac_part):
    # Split time into hours, minutes, and seconds (with fractional part)
    fractional_seconds = float("0." + frac_part)

    # Total seconds since midnight
    total_seconds = hour * 3600 + minute * 60 + second + fractional_seconds
    return total_seconds


# Extract timestamps
def extract_ts(filename):
    substr = filename.split('.')[0]
    hms = substr.split('_')[-2]
    micro = substr.split('_')[-1]
    hh, mm, ss = hms[:2], hms[2:4], hms[4:6]
    # print(hh, mm, ss, micro)
    total_time = ros_time_string_to_seconds(int(hh), int(mm), int(ss), micro)
    # print(total_time)

    return total_time
    #return int(os.path.basename(filename).replace(prefix, "").replace(".png", "").replace(".pcd", ""))

img_ts = [(f, extract_ts(f)) for f in img_files]
print(img_ts[0])
pcd_ts = [(f, extract_ts(f)) for f in pcd_files]
print(pcd_ts[0])


tolerance_ns = 0.50 # 50 milliseconds

matched = []

## When number of images > number of point cloud data
for pcd_file, ts_pcd in pcd_ts:
    # Find closest image within tolerance
    closest = min(img_ts, key=lambda x: abs(x[1] - ts_pcd))
    if abs(closest[1] - ts_pcd) <= tolerance_ns:
        matched.append((closest[0], pcd_file))
        img_ts.remove(closest)  # remove used image


## When number of images < number of point cloud data
# for img_file, ts_img in img_ts:
#     # Find closest point cloud within tolerance
#     closest = min(pcd_ts, key=lambda x: abs(x[1] - ts_img))
#     if abs(closest[1] - ts_img) <= tolerance_ns:
#         matched.append((img_file, closest[0]))
#         pcd_ts.remove(closest)  # remove used point cloud


# print(len(matched))

# Copy matched files to sync folders
index = 0
for img_file, pcd_file in matched:
    # Split filename and extension
    img_name, img_ext = os.path.splitext(os.path.basename(img_file))
    # Format new filename: name_index.ext (e.g., image_0.png)
    new_img_name = f"{img_name}_{index}{img_ext}"
    shutil.copy(img_file, os.path.join(SYNC_IMG_DIR, new_img_name))

    # Keep original PCD filename
    shutil.copy(pcd_file, os.path.join(SYNC_PCD_DIR, os.path.basename(pcd_file)))

    index += 1

print(f"✅ Synced {len(matched)} image-point cloud pairs.")
