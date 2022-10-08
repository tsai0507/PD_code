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
def get_grd_point_set():
    path = './reconstuct_data/camera_path.txt'
    f = open(path, 'r')
    flag=1
    k=[]
    count=0
    for line in f.readlines():
        if(flag==1):
            k.append([])
        if(flag<=2):
            line=line.strip("\n")
            a=float(line)
            (k[count]).append(a)
            flag=1+flag
        elif(flag==3):
            flag=1
            line=line.strip("\n")
            a=float(line)
            (k[count]).append(a)
            count=count+1
    f.close()
    x=k[0][0]
    y=k[0][1]
    z=k[0][2]
    # print(z)
    for i in range(len(k)):
        a=(k[i][0]-x)
        b=(k[i][1]-y)
        c=(k[i][2]-z)
        k[i][0]=b
        k[i][1]=-c
        k[i][2]=a
    return k

def assemble_estimate_path(estimate_path_points,estimate_path_lines,trans):
    point=[0,0,0,1]
    point=trans@point
    point=point[:-1]
    point_ptr=len(estimate_path_points)-1
    estimate_path_points.append(point)
    if (point_ptr>=0):
        temp=[int(point_ptr),int(point_ptr+1)]
        estimate_path_lines.append(temp)

def estimate_path(estimate_path_points,estimate_path_lines):
    colors = [[1, 0, 0] for i in range(len(estimate_path_lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(estimate_path_points)
    # print(estimate_path_points)
    line_set.lines = o3d.utility.Vector2iVector(estimate_path_lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set 

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

    return source_down.transform(result_icp.transformation),result_icp.transformation

### main code ###
print("###start reconstructing###")

### 先初始化存放點雲和存照片的變數 ###
final=[]
pcd_final=o3d.geometry.PointCloud()
target=o3d.geometry.PointCloud()
source=o3d.geometry.PointCloud()
line_set=o3d.geometry.PointCloud()
all_img=[]
#得到所有照片的grd資料
grd_point_set=get_grd_point_set()
grd_point_use=[]
grd_path_use=[]
grd_line_set = o3d.geometry.LineSet()
#用來estimate_path
estimate_path_lines=[]
estimate_path_points=[]
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
        grd_point_use.append(grd_point_set[num_img-1])
        use_img=use_img+1
    elif(use_img==3):
        use_img=0
    else:
        use_img=use_img+1
    NUMBER_IMG=NUMBER_IMG-1
    num_img=num_img+1   
count=len(all_img)
print("Number of img is ",count)
### 做出grd_path ###
for i in range(len(grd_point_use)-1):
    grd_path_use.append([i,i+1])

colors = [[0, 0, 0] for i in range(len(grd_point_use))]
a_line_set = o3d.geometry.LineSet()
a_line_set.points = o3d.utility.Vector3dVector(grd_point_use)
# print(grd_point_set)
a_line_set.colors = o3d.utility.Vector3dVector(colors)
a_line_set.lines = o3d.utility.Vector2iVector(grd_path_use)
grd_line_set=a_line_set


### 若NUMBER_IMG大於兩張,需要做ICP ###
if(count>=2):
    ### 將照片轉為點雲，並且合併 ###

    #先將要作為基準的第一張照片放到final
    count=count-1
    IMG_NUM=0
    target=depth_image_to_point_cloud(all_img[IMG_NUM][0],all_img[IMG_NUM][1])
    IMG_NUM=IMG_NUM+1
    final.append(target)
    SOUR_to_TAR_trsform=np.eye(4)
    assemble_estimate_path(estimate_path_points,estimate_path_lines,SOUR_to_TAR_trsform)
   
    #把轉換過得source放入final且作為下一次的target
    total=count
    while(count!=0):
        count=count-1
        source=depth_image_to_point_cloud(all_img[IMG_NUM][0],all_img[IMG_NUM][1])
        IMG_NUM=IMG_NUM+1
        source, SOUR_to_TAR_trsform =local_icp_algorithm(source,target)
        assemble_estimate_path(estimate_path_points,estimate_path_lines,SOUR_to_TAR_trsform)
        final.append(source)
        target=source
        print("finish : ",int(100-count/total*100),"%")
    
    print("###reconstructing is done###")
    estimate_line_set=estimate_path(estimate_path_points,estimate_path_lines)
    final.append(estimate_line_set)
    final.append(grd_line_set)
    #存pcd資料
    # for point_id in range(len(final)):
    #     pcd_final += final[point_id]
    # pcd_final_down = pcd_final.voxel_down_sample(voxel_size=0.05)
    # o3d.io.write_point_cloud("multiway_registration.pcd", pcd_final_down)
    #顯示結果
    o3d.visualization.draw_geometries(final)

elif(count==1): ### 若NUMBER_IMG 一張,不需要做ICP ###
    target=depth_image_to_point_cloud(all_img[0][0],all_img[0][1])
    final.append(target)
    print("###reconstructing is done###")
    o3d.visualization.draw_geometries(final)
else:
    print("There is not data.")
    print("###reconstructing is done###")

