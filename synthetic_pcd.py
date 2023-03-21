import pickle
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

Rt2T = lambda R,t: np.vstack([np.hstack([R,t]),np.eye(4)[-1,:]])



raw_points = pickle.load(open('Lille1.pickle', 'rb')) #Load Data


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(raw_points[:,:3]) #Load point cloud (column 0:3 are xyz)
N = 50000 #Down-sampling Lidar positions
pos = raw_points[::N,3:6] #(column 0:3 are lidar position xyz)

cameras = []
for p in pos:
  K = np.array([[10,0,5],[0,10,5],[0,0,1]])
  #Fixed rotation angle -50 deg ###################Modify next line######################
  R = Rotation.from_euler('y', -50, degrees=True).as_matrix()@Rotation.from_euler('x', 90, degrees=True).as_matrix()
  #Fixed position (same as existed Lidar position) ###################Modify next line######################
  t = p.reshape((-1,1))
  T = Rt2T(R, -R@t)
  cameras.append(o3d.geometry.LineSet.create_camera_visualization(10,10,K,T))


o3d.visualization.draw_geometries([pcd]+cameras)
