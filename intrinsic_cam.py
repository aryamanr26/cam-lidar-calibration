import cv2
import numpy as np
import glob

# Define checkerboard dimensions (number of **inner corners** per row/column)
## Checkerboard dimensions: 99.85375cm ~ 100cm
# This is a 5x5 checkerboard, meaning it has 4 inner corners in each direction.
CHECKERBOARD = (9, 6)
square_size = 0.047 # Set to your actual square size (e.g., 0.025 meters or 1.0 if arbitrary units)

# 3D points in real world space
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store 3D points (objpoints) and 2D image points (imgpoints)
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Load checkerboard images
FILE_PATH = "data/july8/cam1_cal_data/*.png"
images = glob.glob(FILE_PATH)
images = sorted(images[:50])  # Ensure images are sorted
print(f"Found {len(images)} images to process")

successful_detections = 0
failed_detections = 0

for i, fname in enumerate(images):
    print(f"Processing image {i+1}/{len(images)}: {fname}")
    
    if i % 10 == 0 and i > 0:
        print(f"Progress: {i}/{len(images)} images processed")
        print(f"Successful detections so far: {successful_detections}")
        print(f"Failed detections so far: {failed_detections}")
    
    print(f"  Loading image {i+1}...")
    img = cv2.imread(fname)
    if img is None:
        print(f"  ERROR: Could not load image {fname}")
        failed_detections += 1
        continue
    
    print(f"  Converting to grayscale...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"  Finding checkerboard corners...")
    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        print(f"  ✓ Found corners! Refining...")
        successful_detections += 1
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        print(f"  Drawing corners...")
        # Draw and show corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)
    else:
        print(f"  ✗ No corners found in this image")
        failed_detections += 1

cv2.destroyAllWindows()

print(f"\nFinal Summary:")
print(f"Total images processed: {len(images)}")
print(f"Successful detections: {successful_detections}")
print(f"Failed detections: {failed_detections}")
print(f"Success rate: {(successful_detections/len(images)*100):.1f}%")

if successful_detections < 10:
    print("WARNING: Less than 10 successful detections. Calibration may be unreliable.")
elif successful_detections < 20:
    print("NOTE: Less than 20 successful detections. Consider adding more images for better calibration.")

print("\n=== Starting Camera Calibration ===")
# === Perform Calibration ===
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("✓ Calibration completed successfully!")

# === Output Intrinsic Parameters ===
print("\n=== Calibration Results ===")
print("Camera matrix (intrinsics):\n", cameraMatrix)
print("Distortion coefficients:\n", distCoeffs)
