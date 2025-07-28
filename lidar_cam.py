import cv2
import numpy as np
import open3d as o3d
import glob

# ==== USER PARAMETERS ====
image_path = "data/images/*.png"   # path to checkerboard images
pcd_path = "data/pointclouds/*.pcd" # path to matching pointclouds
fx, fy, cx, cy = 0, 0, 0, 0

checkerboard_dims = (7, 6)    # inner corners (columns, rows)
square_size = 0.025           # meters

# Camera intrinsics (from calibration)
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])
dist_coeffs = np.zeros(5)  # or use actual distortion coeffs

# ==== STEP 1: Compute object points (checkerboard corners in board frame) ====
objp = np.zeros((checkerboard_dims[0]*checkerboard_dims[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
objp *= square_size

# Center the board for better symmetry
objp -= np.mean(objp, axis=0)

# ==== STEP 2: Load matching image and point cloud ====
img_files = sorted(glob.glob(image_path))
pcd_files = sorted(glob.glob(pcd_path))

assert len(img_files) == len(pcd_files), "Mismatch in image/PCD count."

for img_file, pcd_file in zip(img_files, pcd_files):
    print(f"Processing: {img_file} and {pcd_file}")

    # --- Load image and detect checkerboard ---
    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, checkerboard_dims, None)

    if not found:
        print("Checkerboard not found in image.")
        continue

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # --- Solve PnP to get board pose in camera frame ---
    retval, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    R_cam_board, _ = cv2.Rodrigues(rvec)
    t_cam_board = tvec.reshape(3)

    # --- Load point cloud and fit plane ---
    pcd = o3d.io.read_point_cloud(pcd_file)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    board_points = pcd.select_by_index(inliers)

    # --- Estimate checkerboard frame in LiDAR ---
    pts = np.asarray(board_points.points)
    centroid = np.mean(pts, axis=0)

    cov = np.cov((pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1]   # sort largest first
    eigvecs = eigvecs[:, idx]

    z_lidar = np.array([a, b, c])
    z_lidar /= np.linalg.norm(z_lidar)
    x_lidar = eigvecs[:, 0]
    y_lidar = np.cross(z_lidar, x_lidar)
    x_lidar = np.cross(y_lidar, z_lidar)  # ensure orthogonality

    R_board_lidar = np.vstack((x_lidar, y_lidar, z_lidar)).T
    t_board_lidar = centroid.reshape(3)

    # --- Compute LiDAR → Camera transform ---
    R_lidar_cam = R_cam_board @ R_board_lidar.T
    t_lidar_cam = t_cam_board - R_lidar_cam @ t_board_lidar

    T_lidar_cam = np.eye(4)
    T_lidar_cam[:3, :3] = R_lidar_cam
    T_lidar_cam[:3, 3] = t_lidar_cam

    print("\nOptimized Transformation (LiDAR → Camera):")
    print(T_lidar_cam)

    # (Optional) break after first pair
    break
