from bev import *

from re import X
import cv2
import numpy as np
import open3d as o3d

points = []

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)

def To3D(rgb,depth,height,width):
    fov=np.pi/2
    f=0.5*512/(np.tan(fov/2)) #caculate focus length
    #turn piexl coordinate to world  cooridnate    
    point=[]         
    color=[]
    for i in range(width):
        for j in range(height):
            z=depth[i][j][0]/27
            col=rgb[i][j]/255
            point.append([-z*(i-256)/f,z*(j-256)/f,z])  
            color.append([col[0],col[1],col[2]])

    return  [point,color]  

def transfer(pcl):

    #trans bev to front
    rot=np.pi/2
    transform=np.array([[1,0,0,0],
                        [0,np.cos(rot),-np.sin(rot),0],
                        [0,np.sin(rot),np.cos(rot),-1.5],
                        [0,0,0,1]])
    trans_pcl=[]
    for no in pcl:
        no.append(1)
        new=np.dot(transform,no)
        trans_pcl.append([new[0],new[1],new[2]])
    return trans_pcl

def draw_registration_result(source, target, transformation):
    source_temp = source
    target_temp = target
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

front_rgb = "front_view_path.png"
front_depth="front_view_depth.png"
top_rgb = "top_view_path.png"
top_depth="bev_view_depth.png"

RGB = cv2.imread(front_rgb, 1)
DEPTH=cv2.imread(front_depth, 1)
BEV_RGB=cv2.imread(top_rgb, 1)
BEV_DEPTH=cv2.imread(top_depth, 1)

rot=np.pi/2
transform=np.array([[1,0,0,0],
                    [0,np.cos(rot),-np.sin(rot),0],
                    [0,np.sin(rot),np.cos(rot),-1.5],
                    [0,0,0,1]])
# cv2.imshow('image1',RGB)
# cv2.imshow('image2',BEV_RGB)
# cv2.waitKey(100)
# cv2.destroyAllWindows()

cloud1=To3D(RGB,DEPTH,512,512)

cloud=To3D(BEV_RGB,BEV_DEPTH,512,512)
# cloud2=[transfer(cloud[0]),cloud[1]]

pcl2 = o3d.geometry.PointCloud()
pcl2.points = o3d.utility.Vector3dVector(cloud[0])
pcl2.colors = o3d.utility.Vector3dVector(cloud[1])

pcl1 = o3d.geometry.PointCloud()
pcl1.points = o3d.utility.Vector3dVector(cloud1[0])
pcl1.colors = o3d.utility.Vector3dVector(cloud1[1])

threshold = 0.02
evaluation = o3d.pipelines.registration.evaluate_registration(
    pcl1, pcl2, threshold, transform)
print(evaluation)

print("Apply point-to-plane ICP")
reg_p2l = o3d.pipelines.registration.registration_icp(
    pcl1, pcl2, threshold, transform,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2l)
print("Transformation is:")
print(reg_p2l.transformation)
draw_registration_result(pcl1, pcl2, reg_p2l.transformation)


# o3d.visualization.draw_geometries([evaluation],
#                                     zoom=0.4459,
#                                     front=[0.9288, -0.2951, -0.2242],
#                                     lookat=[1.6784, 2.0612, 1.4451],
#                                     up=[-0.3402, -0.9189, -0.1996])

# o3d.visualization.draw_geometries([evaluation])
# o3d.draw_geometries([pcl])
# while(1):
#     o3d.visualization.draw_geometries([pcl])
# numpy_points = np.asarray(pcl.points)
# work on points
# pcl.points = numpy_points

