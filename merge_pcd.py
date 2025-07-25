import open3d as o3d
import numpy as np
import glob

# Load point clouds
pcd1_files = sorted(glob.glob('july8/cam_lidar_data/pointclouds/right/*.pcd'))
pcd2_files = sorted(glob.glob('july8/cam_lidar_data/pointclouds/front/*.pcd'))

print(len(pcd1_files), len(pcd2_files))

for idx in range(len(pcd1_files)):

    # pcd1_path = "july8/cam_lidar_data/pointclouds/right/right_000513_115731_760.pcd"  # Reference lidar
    # pcd2_path = "july8/cam_lidar_data/pointclouds/front/front_000512_115731_720.pcd"  # Lidar to be calibrated

    pcd1_path = pcd1_files[idx]
    pcd2_path = pcd2_files[idx]

    source = o3d.io.read_point_cloud(pcd1_path)  # Moving point cloud
    target = o3d.io.read_point_cloud(pcd2_path)  # Fixed reference

    # Downsample (helps with speed + stability)
    voxel_size = 0.2
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    # Estimate normals for downsampled clouds (needed for FPFH)
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    # Compute FPFH features
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=100))

    # GLOBAL REGISTRATION (RANSAC)
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    # Estimate normals for full-res point clouds (needed for ICP PointToPlane)
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    # LOCAL REFINEMENT (ICP)
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=voxel_size * 1.5,
        init=result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # Print final transformation
    # print("Final Transformation:")
    # print(result_icp.transformation)

    # Apply transformation to source and visualize combined cloud
    source.transform(result_icp.transformation)
    combined = source + target
    # o3d.visualization.draw_geometries([combined])
    o3d.io.write_point_cloud(f"july8/cam_lidar_data/combined_lidar/front-right/combined_{idx}.pcd", combined)

