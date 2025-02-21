import numpy as np
import open3d as o3d
import time

while True:
    try:
        points = np.load("lidar_data/timestep.npy")  # 저장된 데이터를 읽기
        if points.shape[0] > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.visualization.draw_geometries([pcd])
            print(f"Visualizing {len(points)} points")
    except FileNotFoundError:
        print("No Lidar data file found!")

    time.sleep(1)  # 1초마다 업데이트