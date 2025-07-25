import cv2
import numpy as np
import glob

# === CONFIGURATION ===
# Checkerboard settings
checkerboard_dims = (4, 4)  # inner corners per row and column
square_size = 1  # in meters (e.g., 2.5 cm)

# Camera intrinsics
K2 = np.array([[1081.22207031014,    0, 1012.20287739142],
             [0,     1084.61271277629, 628.698761285123],
             [0,                 0,                1]])

D2 = np.zeros(5)  # replace with real distortion if needed

K4 = np.array([[1319.07309918545,     0, 944.889401806983],
              [0,    1324.13643327630, 592.902712685061],
              [0,                  0,                 1]])

D4 = np.zeros(5)

# Image paths (sorted lists of stereo pairs)
imgpaths_cam1 = sorted(glob.glob('july8/cam2_cal_data/*.png'))
imgpaths_cam2 = sorted(glob.glob('july8/cam6_cal_data/*.png'))
print(len(imgpaths_cam1), len(imgpaths_cam2))

assert len(imgpaths_cam1) == len(imgpaths_cam2), "Unequal stereo image count!"

# === PREPARE OBJECT POINTS ===
objp = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
objp *= square_size

# === COLLECT POINTS ===
objpoints = []
imgpoints1 = []
imgpoints2 = []

for img1_path, img2_path in zip(imgpaths_cam1, imgpaths_cam2):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    found1, corners1 = cv2.findChessboardCorners(gray1, checkerboard_dims)
    found2, corners2 = cv2.findChessboardCorners(gray2, checkerboard_dims)

    if found1 and found2:
        objpoints.append(objp)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)

print(f"Collected {len(objpoints)} valid stereo pairs.")

# === CALIBRATE ===
ret, K2, D2, K4, D4, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2,
    K2, D2, K4, D4,
    imageSize=gray1.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    flags=cv2.CALIB_FIX_INTRINSIC
)

# === OUTPUT ===
print("\nRotation (Cam2 wrt Cam1):\n", R)
print("\nTranslation (Cam2 wrt Cam1):\n", T)

T_21 = np.eye(4)
T_21[:3, :3] = R
T_21[:3, 3] = T.flatten()
print("\nFull 4x4 transformation matrix:\n", T_21)
