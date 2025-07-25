# # Load point clouds
# pcd_a = o3d.io.read_point_cloud("july8/cam_lidar_data/pointclouds/front/front_000119_115652_419.pcd") #
# pcd_b = o3d.io.read_point_cloud("july8/cam_lidar_data/pointclouds/left/left_000120_115652_497.pcd")

# # Downsample
# pcd_a = pcd_a.voxel_down_sample(0.1)
# pcd_b = pcd_b.voxel_down_sample(0.1)

# # Initial guess (identity or based on rough mount)
# init_guess = np.eye(4)

# # ICP
# threshold = 1.5  # in meters
# reg = o3d.pipelines.registration.registration_icp(
#     pcd_a, pcd_b, threshold, init_guess,
#     o3d.pipelines.registration.TransformationEstimationPointToPoint()
# )

# print("Extrinsic (A to B):")
# print(reg.transformation)

#!/usr/bin/env python3
"""
Lidar-to-Lidar Calibration Script
Performs calibration between two time-synchronized lidar point clouds
"""

import numpy as np
import open3d as o3d
import copy
import time
from typing import Tuple, Optional
import matplotlib.pyplot as plt

class LidarCalibration:
    def __init__(self, voxel_size: float = 0.1):
        """
        Initialize the calibration object
        
        Args:
            voxel_size: Voxel size for downsampling and feature extraction
        """
        self.voxel_size = voxel_size
        self.transformation_matrix = np.eye(4)
        
    def load_point_clouds(self, pcd1_path: str, pcd2_path: str) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        Load two point cloud files
        
        Args:
            pcd1_path: Path to first point cloud (reference)
            pcd2_path: Path to second point cloud (to be transformed)
            
        Returns:
            Tuple of loaded point clouds
        """
        pcd1 = o3d.io.read_point_cloud(pcd1_path)
        pcd2 = o3d.io.read_point_cloud(pcd2_path)
        
        print(f"Loaded PCD1: {len(pcd1.points)} points")
        print(f"Loaded PCD2: {len(pcd2.points)} points")
        
        return pcd1, pcd2
    
    def preprocess_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Preprocess point cloud: downsample, remove outliers, estimate normals
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Preprocessed point cloud
        """
        # Downsample
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        
        # Remove outliers
        pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Estimate normals
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2, max_nn=30
            )
        )
        
        return pcd_down
    
    def extract_fpfh_features(self, pcd: o3d.geometry.PointCloud) -> o3d.pipelines.registration.Feature:
        """
        Extract FPFH (Fast Point Feature Histograms) features
        
        Args:
            pcd: Input point cloud with normals
            
        Returns:
            FPFH features
        """
        return o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 5, max_nn=100
            )
        )
    
    def global_registration(self, pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Perform global registration using RANSAC
        
        Args:
            pcd1: Reference point cloud
            pcd2: Point cloud to be aligned
            
        Returns:
            Transformation matrix
        """
        # Extract features
        fpfh1 = self.extract_fpfh_features(pcd1)
        fpfh2 = self.extract_fpfh_features(pcd2)
        
        # RANSAC registration
        distance_threshold = self.voxel_size * 1.5
        
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd1, pcd2, fpfh1, fpfh2,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        return result.transformation
    
    def local_registration_icp(self, pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud, 
                              init_transform: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Perform local refinement using ICP
        
        Args:
            pcd1: Reference point cloud
            pcd2: Point cloud to be aligned
            init_transform: Initial transformation from global registration
            
        Returns:
            Refined transformation matrix and fitness score
        """
        # Point-to-plane ICP
        result = o3d.pipelines.registration.registration_icp(
            pcd1, pcd2, self.voxel_size * 0.4, init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        
        return result.transformation, result.fitness
    
    def colored_icp(self, pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud, 
                   init_transform: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Perform colored ICP if point clouds have color/intensity information
        
        Args:
            pcd1: Reference point cloud
            pcd2: Point cloud to be aligned
            init_transform: Initial transformation
            
        Returns:
            Refined transformation matrix and fitness score
        """
        result = o3d.pipelines.registration.registration_colored_icp(
            pcd1, pcd2, self.voxel_size, init_transform,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
            )
        )
        
        return result.transformation, result.fitness
    
    def evaluate_registration(self, pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud, 
                            transformation: np.ndarray) -> dict:
        """
        Evaluate registration quality
        
        Args:
            pcd1: Reference point cloud
            pcd2: Transformed point cloud
            transformation: Transformation matrix
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Apply transformation
        pcd2_transformed = copy.deepcopy(pcd2)
        pcd2_transformed.transform(transformation)
        
        # Calculate registration metrics
        evaluation = o3d.pipelines.registration.evaluate_registration(
            pcd1, pcd2_transformed, self.voxel_size * 0.4, transformation
        )
        
        return {
            'fitness': evaluation.fitness,
            'inlier_rmse': evaluation.inlier_rmse,
            'correspondence_set': len(evaluation.correspondence_set)
        }
    
    def visualize_registration(self, pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud, 
                             transformation: np.ndarray, title: str = "Registration Result"):
        """
        Visualize the registration result
        
        Args:
            pcd1: Reference point cloud
            pcd2: Point cloud to be aligned
            transformation: Transformation matrix
            title: Window title
        """
        pcd1_vis = copy.deepcopy(pcd1)
        pcd2_vis = copy.deepcopy(pcd2)
        
        # Color the point clouds
        pcd1_vis.paint_uniform_color([1, 0, 0])  # Red
        pcd2_vis.paint_uniform_color([0, 1, 0])  # Green
        
        # Transform second point cloud
        pcd2_vis.transform(transformation)
        
        # Visualize
        o3d.visualization.draw_geometries([pcd1_vis, pcd2_vis], window_name=title)
    
    def calibrate(self, pcd1_path: str, pcd2_path: str, visualize: bool = True) -> dict:
        """
        Main calibration function
        
        Args:
            pcd1_path: Path to first point cloud (reference)
            pcd2_path: Path to second point cloud (to be transformed)
            visualize: Whether to show visualization
            
        Returns:
            Dictionary with calibration results
        """
        print("Starting lidar-to-lidar calibration...")
        
        # Load point clouds
        pcd1, pcd2 = self.load_point_clouds(pcd1_path, pcd2_path)
        
        if visualize:
            print("Showing original point clouds...")
            self.visualize_registration(pcd1, pcd2, np.eye(4), "Original Point Clouds")
        
        # Preprocess
        print("Preprocessing point clouds...")
        pcd1_processed = self.preprocess_point_cloud(pcd1)
        pcd2_processed = self.preprocess_point_cloud(pcd2)
        
        # Global registration
        print("Performing global registration...")
        start_time = time.time()
        global_transform = self.global_registration(pcd1_processed, pcd2_processed)
        global_time = time.time() - start_time
        
        print(f"Global registration completed in {global_time:.2f} seconds")
        
        # Evaluate global registration
        global_eval = self.evaluate_registration(pcd1_processed, pcd2_processed, global_transform)
        print(f"Global registration - Fitness: {global_eval['fitness']:.4f}, RMSE: {global_eval['inlier_rmse']:.4f}")
        
        if visualize:
            print("Showing global registration result...")
            self.visualize_registration(pcd1, pcd2, global_transform, "Global Registration")
        
        # Local refinement with ICP
        print("Performing local refinement with ICP...")
        start_time = time.time()
        local_transform, fitness = self.local_registration_icp(pcd1_processed, pcd2_processed, global_transform)
        local_time = time.time() - start_time
        
        print(f"Local refinement completed in {local_time:.2f} seconds")
        
        # Evaluate local registration
        local_eval = self.evaluate_registration(pcd1_processed, pcd2_processed, local_transform)
        print(f"Local registration - Fitness: {local_eval['fitness']:.4f}, RMSE: {local_eval['inlier_rmse']:.4f}")
        
        if visualize:
            print("Showing final registration result...")
            self.visualize_registration(pcd1, pcd2, local_transform, "Final Registration")
        
        # Store final transformation
        self.transformation_matrix = local_transform
        
        # Prepare results
        results = {
            'transformation_matrix': local_transform,
            'global_registration': {
                'transformation': global_transform,
                'evaluation': global_eval,
                'time': global_time
            },
            'local_registration': {
                'transformation': local_transform,
                'evaluation': local_eval,
                'time': local_time
            },
            'rotation_matrix': local_transform[:3, :3],
            'translation_vector': local_transform[:3, 3],
            'total_time': global_time + local_time
        }
        
        return results
    
    def save_transformation(self, filepath: str):
        """
        Save transformation matrix to file
        
        Args:
            filepath: Path to save the transformation matrix
        """
        np.save(filepath, self.transformation_matrix)
        print(f"Transformation matrix saved to {filepath}")
    
    def load_transformation(self, filepath: str):
        """
        Load transformation matrix from file
        
        Args:
            filepath: Path to load the transformation matrix
        """
        self.transformation_matrix = np.load(filepath)
        print(f"Transformation matrix loaded from {filepath}")
    
    def apply_transformation(self, pcd_path: str, output_path: str):
        """
        Apply the calibrated transformation to a point cloud
        
        Args:
            pcd_path: Path to input point cloud
            output_path: Path to save transformed point cloud
        """
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd.transform(self.transformation_matrix)
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Transformed point cloud saved to {output_path}")

def main():
    """
    Example usage of the LidarCalibration class
    """
    # Initialize calibration object
    calibrator = LidarCalibration(voxel_size=0.1)
    
    print("Calibrating Right to Front LiDAR")
    # Paths to your point cloud files
    pcd1_path = "july8/cam_lidar_data/pointclouds/right/right_000513_115731_760.pcd"  # Reference lidar
    pcd2_path = "july8/cam_lidar_data/pointclouds/front/front_000512_115731_720.pcd"  # Lidar to be calibrated
    
    try:
        # Perform calibration
        results = calibrator.calibrate(pcd1_path, pcd2_path, visualize=True)
        
        # Print results
        print("\n" + "="*50)
        print("CALIBRATION RESULTS")
        print("="*50)
        print(f"Total processing time: {results['total_time']:.2f} seconds")
        print(f"Final fitness score: {results['local_registration']['evaluation']['fitness']:.4f}")
        print(f"Final RMSE: {results['local_registration']['evaluation']['inlier_rmse']:.4f}")
        
        print("\nTransformation Matrix:")
        print(results['transformation_matrix'])
        
        print("\nRotation Matrix:")
        print(results['rotation_matrix'])
        
        print("\nTranslation Vector:")
        print(results['translation_vector'])
        
        # Save transformation matrix
        calibrator.save_transformation("lidar_calibration_transform.npy")
        
        # Example: Apply transformation to new data
        # calibrator.apply_transformation("new_lidar2_data.pcd", "transformed_lidar2_data.pcd")
        
    except FileNotFoundError:
        print("Error: Point cloud files not found!")
        print("Please update the file paths in the script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()