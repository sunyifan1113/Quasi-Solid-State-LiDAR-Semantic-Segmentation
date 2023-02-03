import glob
import rosbag #pip3 install bagpy
import pcl #apt install python3-pcl
import numpy as np
import sensor_msgs.point_cloud2 as PC2

PointCloud2Pcd = lambda msg: np.array([p for p in PC2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))]) #see msg.fields
PointCloud2PcdXYZI = lambda msg: np.array([p for p in PC2.read_points(msg, skip_nans=True, field_names=("x", "y","z","intensity"))]) #see msg.fields
def readBag(bagPath, topics, data=dict()):
    for topic, msg, t in rosbag.Bag(bagPath).read_messages():
        if topic in topics:
            if topic not in data: data[topic] = []
            data[topic].append((t,msg))
    return data

lidarTopics = ['/cepton/points']

topics = lidarTopics
bagFiles = sorted(glob.glob("*.bag"))
    
for bag in bagFiles:
  data = readBag(bag, topics)
  for i in range(len(lidarTopics)):
    msg = data[lidarTopics[i]][-1][1] #!!!!!!!!!!!!!!Export the last frame only
    pcd = pcl.PointCloud_PointXYZI()
    pcd.from_array(PointCloud2PcdXYZI(msg).astype(np.float32))
    pcl.save(pcd,bag[:-4]+'.pcd')
