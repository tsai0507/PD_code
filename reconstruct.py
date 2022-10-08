import open3d as o3d
import copy
import os 
import cv2
import numpy as np
# from PIL import Image

def depth_image_to_point_cloud(rgb,depth):
    height=512
    width=512
    fov=np.pi/2
    f=0.5*512/(np.tan(fov/2)) #caculate focus length
    #turn piexl coordinate to world  cooridnate    
    point=[]         
    color=[]
    for i in range(width):
        for j in range(height):
            z=depth[i][j][0]/25.5
            col=rgb[i][j]/255
            point.append([z*(i-256)/f,z*(j-256)/f,z])  
            color.append([col[0],col[1],col[2]])
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(point)
    pcl.colors = o3d.utility.Vector3dVector(color)
    pcl.estimate_normals()

    return  pcl

def preprocess_point_cloud(pcd, voxel_size):
   
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size,source,target):

    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh 

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size,result_ransac):
    distance_threshold = voxel_size * 0.4

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def local_icp_algorithm(source,target):
    voxel_size = 0.05  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size,source,target)

    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,voxel_size,result_ransac)

    return source.transform(result_icp.transformation)

### main code ###
print("###start reconstructing###")

### 先初始化存放點雲和存照片的變數 ###
final=[]
target=o3d.geometry.PointCloud()
source=o3d.geometry.PointCloud()
all_img=[]

### 將rgb,depth資料讀入並且存成list ###
DIR = './reconstuct_data' #要統計的資料夾
NUMBER_IMG=len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])/2
num_img=1
use_img=0
while(NUMBER_IMG>0):
    if(use_img==0):
        img0=cv2.imread('./reconstuct_data/'+'rgb_'+str(num_img)+'.png',1)
        img1=cv2.imread('./reconstuct_data/'+'img1_depth'+str(num_img)+'.png',1)
        temp=(img0,img1)
        all_img.append(temp)
        use_img=use_img+1
    elif(use_img==2):
        use_img=0
    else:
        use_img=use_img+1
    NUMBER_IMG=NUMBER_IMG-1
    num_img=num_img+1   
count=len(all_img)
print("Number of img is ",count)

### 若NUMBER_IMG大於兩張,需要做ICP ###
if(count>=2):
    ### 將照片轉為點雲，並且合併 ###
    #先將要作為基準的第一張照片放到final
    count=count-1
    IMG_NUM=0
    target=depth_image_to_point_cloud(all_img[IMG_NUM][0],all_img[IMG_NUM][1])
    IMG_NUM=IMG_NUM+1
    final.append(target)
    #把轉換過得source放入final且作為下一次的target
    total=count
    while(count!=0):
        count=count-1
        source=depth_image_to_point_cloud(all_img[IMG_NUM][0],all_img[IMG_NUM][1])
        IMG_NUM=IMG_NUM+1
        source=local_icp_algorithm(source,target)
        final.append(source)
        target=source
        print("finish : ",int(100-count/total*100),"%")
    print("###reconstructing is done###")
    o3d.visualization.draw_geometries(final)
elif(count==1): ### 若NUMBER_IMG 一張,不需要做ICP ###
    target=depth_image_to_point_cloud(all_img[0][0],all_img[0][1])
    final.append(target)
    print("###reconstructing is done###")
    o3d.visualization.draw_geometries(final)
else:
    print("There is not data.")
    print("###reconstructing is done###")

