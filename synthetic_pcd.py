import pickle
from math import atan, degrees
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

Rt2T = lambda R,t: np.vstack([np.hstack([R,t]),np.eye(4)[-1,:]])

def angle_between_points(y1, y2, x1, x2):
  if x2-x1 == 0:
    if y2 > y1:
      return 90
    elif y2 < y1:
      return -90
    else:
      return None
  return degrees(atan((y2-y1)/(x2-x1)))

raw_points = pickle.load(open('Lille1.pickle', 'rb')) #Load Data


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(raw_points[::500,:3]) #Load point cloud (column 0:3 are xyz)
N = 50000 #Down-sampling Lidar positions
pos = raw_points[::N,3:7] #(column 3:6 are lidar position xyz, column 6 is the timestamp)
pos = sorted(pos, key=lambda pos:pos[3]) #sort pos with increasing timestamp
radius = 60 #radius of visibility
angle_of_visibility = 90 #range of visibility in degrees from center
visible_idx = set() #set of indexes of point clouds that are visible

cameras = []
for i in range(1, len(pos)):
  _, pt_map = pcd.hidden_point_removal(pos[i][:3], radius)
  for idx in pt_map:
    if -angle_of_visibility/2 <= angle_between_points(pcd.points[idx][1], pos[i][1], pcd.points[idx][0], pos[i][0]) <= angle_of_visibility/2:
      visible_idx.add(idx)
  K = np.array([[10,0,5],[0,10,5],[0,0,1]])
  rotation_angle = angle_between_points(0, pos[i-1][1], 0, pos[i-1][0])
  R = Rotation.from_euler('y', 90+rotation_angle, degrees=True).as_matrix()@Rotation.from_euler('x', 90, degrees=True).as_matrix()
  #Fixed position (same as existed Lidar position) ###################Modify next line######################
  t = pos[i][0:3].reshape((-1,1))
  T = Rt2T(R, -R@t)
  cameras.append(o3d.geometry.LineSet.create_camera_visualization(10,10,K,T))

print('number of points in raw dataset:', len(pcd.points))
pcd = pcd.select_by_index(list(visible_idx))
print('number of points visible:', len(pcd.points))
o3d.visualization.draw_geometries([pcd]+cameras[::2])
