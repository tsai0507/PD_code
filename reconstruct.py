from data_collect import *
import open3d as o3d
import copy

print("start")
final=[]
target=o3d.geometry.PointCloud()
source=o3d.geometry.PointCloud()
all_img=[]

def To3Dcloud(rgb,depth):
    height=512
    width=512
    fov=np.pi/2
    f=0.5*512/(np.tan(fov/2)) #caculate focus length
    #turn piexl coordinate to world  cooridnate    
    point=[]         
    color=[]
    for i in range(width):
        for j in range(height):
            z=depth[i][j]/25.5
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

def pcl_merge(source,target):
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


action = "move_forward"
img =navigateAndSee(action)
all_img.append(img)
while True:
    keystroke = cv2.waitKey(0) #等待按鍵事件
    if keystroke == ord(FORWARD_KEY): #ord()取得char得ASCII
        action = "move_forward"
        img =navigateAndSee(action)
        all_img.append(img)
        print("action: FORWARD")
    elif keystroke == ord(LEFT_KEY):
        action = "turn_left"
        img =navigateAndSee(action)
        all_img.append(img)
        print("action: LEFT")
    elif keystroke == ord(RIGHT_KEY):
        action = "turn_right"
        img =navigateAndSee(action)
        all_img.append(img)
        print("action: RIGHT")
    elif keystroke == ord(FINISH):
        print("action: FINISH ")
        break
    else:
        print("INVALID KEY")
        continue

count=len(all_img)
if(count>2):
    count=count-1
    IMG_NUM=0
    target=To3Dcloud(all_img[IMG_NUM][0],all_img[IMG_NUM][1])
    IMG_NUM=IMG_NUM+1
    final.append(target)
    while(count!=0):
        count=count-1
        
        source=To3Dcloud(all_img[IMG_NUM][0],all_img[IMG_NUM][1])
        IMG_NUM=IMG_NUM+1
        source=pcl_merge(source,target)
        final.append(source)
        target=source



o3d.visualization.draw_geometries(final)