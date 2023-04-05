import pickle
from math import atan, degrees
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

Rt2T = lambda R,t: np.vstack([np.hstack([R,t]),np.eye(4)[-1,:]])
normalizedLastRow = lambda X: np.divide(X,X[-1,:])
proj = lambda K,T,X: normalizedLastRow(K@T[:3,:]@X)
hom = lambda x: np.vstack([x,np.ones((1,x.shape[1]))])
addZero = lambda x: np.vstack([x,np.zeros((1,x.shape[1]))])


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
pattern,pattern_X = pickle.load(open('pattern.pickle', 'rb')) #Load scnning pattern


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(raw_points[::500,:3]) #Load point cloud (column 0:3 are xyz)
N = 50000 #Down-sampling Lidar positions
pos = raw_points[::N,3:7] #(column 3:6 are lidar position xyz, column 6 is the timestamp)
pos = sorted(pos, key=lambda pos:pos[3]) #sort pos with increasing timestamp
radius = 60 #radius of visibility
angle_of_visibility = 90 #range of visibility in degrees from center
visible_idx = set() #set of indexes of point clouds that are visible

cameras = []
T_cam = []
for i in range(1, len(pos)): #test on one frame
  #_, pt_map = pcd.hidden_point_removal(pos[i][:3], radius) #Wrong API Call, parameters are incorrect
  '''
  for idx in pt_map:
    if -angle_of_visibility/2 <= angle_between_points(pcd.points[idx][1], pos[i][1], pcd.points[idx][0], pos[i][0]) <= angle_of_visibility/2:
      visible_idx.add(idx)
  '''
  K = np.array([[10,0,5],[0,10,5],[0,0,1]])
  heading_angle = angle_between_points(pos[i][1], pos[i-1][1], pos[i][0], pos[i-1][0])
  R = Rotation.from_euler('y', 90+heading_angle, degrees=True).as_matrix()@Rotation.from_euler('x', 90, degrees=True).as_matrix()
  #Fixed position (same as existed Lidar position) ###################Modify next line######################
  t = pos[i][0:3].reshape((-1,1))
  T = Rt2T(R, -R@t)
  T_cam.append(T)
  cameras.append(o3d.geometry.LineSet.create_camera_visualization(10,10,K,T))

print('number of points in raw dataset:', len(pcd.points))
pcd = pcd.select_by_index(list(visible_idx))
print('number of points visible:', len(pcd.points))
o3d.visualization.draw_geometries([pcd]+cameras[::2])


K = np.array([[1000,0,1200],[0,1000,500],[0,0,1]])
w = 2350
h = 760
s = 5

#for i in range(1, len(pos)):
for i in range(1, 2): #Test on one frame
  p = pos[i]
  ########################Select points in camera FoV########################
  cropped_raw_points = raw_points[np.where(np.logical_and(raw_points[:,4]>p[1]-20, raw_points[:,4]<p[1]+40))[0],:]
  pcd.points = o3d.utility.Vector3dVector(cropped_raw_points[:,:3]) #Load point cloud (column 0:3 are xyz)
  ########################Project to camera frame########################
  X = hom(cropped_raw_points[:,:3].T)
  T = T_cam[i]
  X1 = T@X
  X1 = X1[:,np.where(X1[2]>0)[0]] #keep points in front of camera
  x = proj(K,np.eye(4),X1)
  idx = np.where(np.logical_and(np.logical_and(x[0,:]>0, x[0,:]<w),np.logical_and(x[1,:]>0, x[1,:]<h)))[0] #keep points in camera FoV

  #pcd.points = o3d.utility.Vector3dVector(X1[:-1,:].T)
  #o3d.visualization.draw_geometries([pcd])

  #plt.scatter(x[0,idx],x[1,idx],c=X1[2,idx].astype(np.int32),s=1)
  #plt.show()

  ########################Convert to depth image with label and class########################
  depth_img = np.zeros((int(h/s),int(w/s),3))
  depth_img_raw = np.array([x[0,idx], x[1,idx], X1[2,idx], cropped_raw_points[idx,8], cropped_raw_points[idx,9]]) #u,v,depth,label,class
  depth_img_raw = depth_img_raw[:,(-depth_img_raw[2,:]).argsort()] #sort by depth
  idx_u = (depth_img_raw[1,:]/s).astype(np.int32) #scale by s
  idx_v = (depth_img_raw[0,:]/s).astype(np.int32)
  depth_img[idx_u,idx_v,0] = 1/depth_img_raw[2,:] #load inverse of depth
  depth_img[idx_u,idx_v,1] = depth_img_raw[3,:] #load labels
  depth_img[idx_u,idx_v,2] = depth_img_raw[4,:] #load class
  #plt.imshow(depth_img[:,:,0])
  #plt.show()

  ########################Sample from depth image########################
  idx_uv = (pattern[0][[1,0],:]/s).astype(np.int32)
  sampledPointInfo = depth_img[idx_uv[0],idx_uv[1],:] #1/d, label, class
  valid_idx = np.where(sampledPointInfo[:,0]>0)[0]
  X2 = np.multiply(np.linalg.inv(K)@pattern[0][:,valid_idx],1/sampledPointInfo[valid_idx,0])
  #pcd.points = o3d.utility.Vector3dVector(X2.T)
  #o3d.visualization.draw_geometries([pcd])
  output = np.block([X2.T,sampledPointInfo[valid_idx,1:]]) #x, y, z, label, class
