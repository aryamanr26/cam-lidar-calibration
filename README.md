# cam-lidar-calibration
Calibration pipeline for a multi-sensor rig (6 cameras, 4 LiDARs) with support for both intrinsic and extrinsic calibration and data extraction.

### Calibration Pipeline Steps

1. **Download the Data**  
   Download the dataset from AWS S3 to your local machine.

2. **Extract Sensor Data**  
   Use one of the following scripts to convert the `.mcap` data:
   - (Preferred) `all_data_extractor.py`: Extracts both image (`.png`) and point cloud (`.pcd`) data.
   - `bag_extractor.py`: Same as the above script but hardcoded for a few values only.

3. **Compute Camera Intrinsics**  
   - Run `intrinsic_cam.py`  
   **OR**  
   - (Preferred) Use the **MATLAB Camera Calibration Toolbox** for more flexibility and save the intrinsics as a `.mat` file.

4. **Camera-LiDAR Time Synchronization**  
   - Run `time_sync.py` to ensure equal number of samples for both camera and LiDAR data. Both data streams have different frequencies.

5. **Upload to MATLAB (if using MATLAB Online)**  
   - Upload the synced `.png` and `.pcd` files to [MATLAB Online](https://matlab.mathworks.com/).  
   *(Skip this step if running MATLAB locally.)*

6. **Configure Calibration Tool**  
   - Provide the following inputs:
     - `image_path`
     - `pcd_path`
     - Checkerboard size
     - Camera intrinsics (preferably in `.mat` format)

7. **Initial Detection Fails? Adjust ROI & Plane**  
   - If checkerboard detection fails initially (common with this dataset), manually:
     - Adjust the **ROI**
     - Select the **calibration plane**
     - Pick **checkerboard corners**
   - You will then be prompted to tweak threshold settings.

8. **Run Calibration**  
   - Press the **Detect** button again  
   - Then hit **Calibrate**

9. **Save Results**  
   - Save the calibration variables (camera intrinsics & extrinsics) into a `.mat` file for later use.

10. **Reprojection**
   - Will soon update.
