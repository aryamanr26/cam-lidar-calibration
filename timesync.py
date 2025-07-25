import os
import glob
import shutil

## CAHNGE ALL DIRECTORIES BEFORE RUNNING
folder = "cam_lidar_data"
IMG_DIR = f"july8/{folder}/images/cam6"
PCD_DIR = f"july8/{folder}/pointclouds/front"  # Update which LIDAR you are using
SYNC_IMG_DIR = f"july8/{folder}/synced/cam6/images"
SYNC_PCD_DIR = f"july8/{folder}/synced/cam6/frontpointclouds"

# /home/mcity/Downloads/mcap_extraction/cam2_cal_data/images/image_000000_134339_018.png

# camera 2 uses front lidar
# camera 4 uses right lidar
# camera 6 uses both front and right lidars

# Camera - LIDAR Config
#      1     2
#   5     F     6
#   3   L   R   4
#         B
#

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
for img_file, pcd_file in matched:
    #shutil.copy(img_file, os.path.join(SYNC_IMG_DIR, os.path.basename(img_file)))
    shutil.copy(pcd_file, os.path.join(SYNC_PCD_DIR, os.path.basename(pcd_file)))

print(f"✅ Synced {len(matched)} image-point cloud pairs.")
