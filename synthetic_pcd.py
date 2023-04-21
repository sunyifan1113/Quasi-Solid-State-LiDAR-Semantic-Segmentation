import pickle
import os
from math import atan, degrees
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from sklearn.model_selection import train_test_split

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

label_map = {
  000000000.0:0,
  100000000.0:1,
  200000000.0:2,
  201000000.0:3,
  202000000.0:4,
  202010000.0:5,
  202020000.0:6,
  202030000.0:7,
  202040000.0:8,
  202050000.0:9,
  202060000.0:10,
  203000000.0:11,
  300000000.0:12,
  301000000.0:13,
  302000000.0:14,
  302010000.0:15,
  302020000.0:16,
  302020100.0:17,
  302020200.0:18,
  302020300.0:19,
  302020400.0:20,
  302020500.0:21,
  302020600.0:22,
  302020700.0:23,
  302020800.0:24,
  302020900.0:25,
  302021000.0:26,
  302021100.0:27,
  302030000.0:28,
  302030100.0:29,
  302030200.0:30,
  302030300.0:31,
  302030400.0:32,
  302030500.0:33,
  302030600.0:34,
  302030700.0:35,
  302040000.0:36,
  302040100.0:37,
  302040200.0:38,
  302040300.0:39,
  302040400.0:40,
  302040500.0:41,
  303000000.0:42,
  303010000.0:43,
  303020000.0:44,
  303020100.0:45,
  303020200.0:46,
  303020300.0:47,
  303020400.0:48,
  303020500.0:49,
  303020600.0:50,
  303020700.0:51,
  303020800.0:52,
  303020900.0:53,
  303021000.0:54,
  303030000.0:55,
  303030100.0:56,
  303030200.0:57,
  303030201.0:58,
  303030202.0:59,
  303030203.0:60,
  303030300.0:61,
  303030301.0:62,
  303030302.0:63,
  303030303.0:64,
  303030400.0:65,
  303030401.0:66,
  303030402.0:67,
  303030403.0:68,
  303030500.0:69,
  303030501.0:70,
  303030502.0:71,
  303030503.0:72,
  303040000.0:73,
  303040100.0:74,
  303040200.0:75,
  303040201.0:76,
  303040202.0:77,
  303040203.0:78,
  303040204.0:79,
  303040205.0:80,
  303040206.0:81,
  303040207.0:82,
  303040208.0:83,
  303040209.0:84,
  303040300.0:85,
  303040400.0:86,
  303040500.0:87,
  303050000.0:88,
  303050100.0:89,
  303050200.0:90,
  303050300.0:91,
  303050400.0:92,
  303050500.0:93,
  303050600.0:94,
  304000000.0:95,
  304010000.0:96,
  304020000.0:97,
  304030000.0:98,
  304040000.0:99,
  304050000.0:100
}
def remove_zeros(list):
  res = []
  for n in list:
    s = str(n)
    o = ''
    for i in range(len(s)):
      if s[i] != '0':
        o += s[i]
    o = o[:-1]
    if o == '':
      o = '0'
    res.append(int(o))
  return res


raw_points = pickle.load(open('Paris.pickle', 'rb')) #Load Data
pattern,pattern_X = pickle.load(open('pattern.pickle', 'rb')) #Load scnning pattern

current_path = os.getcwd()
point_cloud_path = os.path.join(current_path, 'Lille_point_cloud')
if os.path.exists(point_cloud_path):
    pass
else:
    os.makedirs(point_cloud_path)
    os.makedirs(os.path.join(point_cloud_path, 'train'))
    os.makedirs(os.path.join(point_cloud_path, 'test'))
label_path = os.path.join(current_path, 'Lille_label')
if os.path.exists(label_path):
    pass
else:
    os.makedirs(label_path)
    os.makedirs(os.path.join(label_path, 'train'))
    os.makedirs(os.path.join(label_path, 'test'))


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(raw_points[:,:3]) #Load point cloud (column 0:3 are xyz)
N = 50000 #Down-sampling Lidar positions
pos = raw_points[::N,3:7] #(column 3:6 are lidar position xyz, column 6 is the timestamp)
pos = sorted(pos, key=lambda pos:pos[3]) #sort pos with increasing timestamp
radius = 60 #radius of visibility
angle_of_visibility = 90 #range of visibility in degrees from center
visible_idx = set() #set of indexes of point clouds that are visible
print(len(pos))

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

'''
print('number of points in raw dataset:', len(pcd.points))
pcd = pcd.select_by_index(list(visible_idx))
print('number of points visible:', len(pcd.points))
o3d.visualization.draw_geometries([pcd]+cameras[::2])
'''

K = np.array([[1000,0,1200],[0,1000,500],[0,0,1]])
w = 2350
h = 760
s = 5

train_test_idx = train_test_split(range(1,len(pos)), test_size=0.2, random_state=42)

for i in range(1, len(pos)):
#for i in range(1, 2): #Test on one frame
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
  depth_img = np.zeros((int(h/s),int(w/s),4))
  depth_img_raw = np.array([x[0,idx], x[1,idx], X1[2,idx], cropped_raw_points[idx,7], cropped_raw_points[idx,8], cropped_raw_points[idx,9]]) #u,v,depth,intensity,label,class
  depth_img_raw = depth_img_raw[:,(-depth_img_raw[2,:]).argsort()] #sort by depth
  idx_u = (depth_img_raw[1,:]/s).astype(np.int32) #scale by s
  idx_v = (depth_img_raw[0,:]/s).astype(np.int32)
  depth_img[idx_u,idx_v,0] = 1/depth_img_raw[2,:] #load inverse of depth
  depth_img[idx_u,idx_v,1] = depth_img_raw[3,:]/255 #load intensity
  depth_img[idx_u,idx_v,2] = depth_img_raw[4,:] #load labels
  depth_img[idx_u,idx_v,3] = depth_img_raw[5,:] #load class
  #plt.imshow(depth_img[:,:,0])
  #plt.show()

  ########################Sample from depth image########################
  idx_uv = (pattern[0][[1,0],:]/s).astype(np.int32)
  sampledPointInfo = depth_img[idx_uv[0],idx_uv[1],:] #1/d, intensity, label, class
  valid_idx = np.where(sampledPointInfo[:,0]>0)[0]
  X2 = np.multiply(np.linalg.inv(K)@pattern[0][:,valid_idx],1/sampledPointInfo[valid_idx,0])
  X2 = np.linalg.inv(T)@X2
  #pcd.points = o3d.utility.Vector3dVector(X2.T)
  #o3d.visualization.draw_geometries([pcd])
  output = np.block([X2.T,sampledPointInfo[valid_idx,1:]]) #x, y, z, intensity, label, class
  output_stack = np.block([output1,output2])
  filename = str(i+1961).zfill(6)
  bin_data = np.array(output[:,:4]).astype(np.float32)
  label_data = np.array([label_map[k] if k in label_map else 0 for k in output[:,5]]).reshape(-1).astype(np.int32)
  #print(label_data)
  if i in train_test_idx[1]:
    filename = os.path.join('test', filename)
  else:
    filename = os.path.join('train', filename)
  bin_file = os.path.join(point_cloud_path, filename) + '.bin'
  label_file = os.path.join(label_path, filename) + '.label'
  bin_data.tofile(bin_file)
  label_data.tofile(label_file)
  print(i+1961)
  

