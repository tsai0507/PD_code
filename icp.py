import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import cv2
import copy
import math


# def output_transform(A, B):
#     '''
#     Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
#     Input:
#       A: Nxm numpy array of corresponding points
#       B: Nxm numpy array of corresponding points
#     Returns:
#       T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
#       R: mxm rotation matrix
#       t: mx1 translation vector
#     '''


#     # get number of dimensions
#     m=(A.point[0]).shape

#     # translate points to their centroids
#     centroid_A = A.get_center()
#     print(centroid_A)
#     centroid_B = B.get_center()
#     AA = A.shape - centroid_A
#     BB = B.shape - centroid_B
#     # rotation matrix
#     W = np.dot(AA.T, BB)
#     U, S, Vt = np.linalg.svd(W)
#     R = np.dot(Vt.T, U.T)

#     # # special reflection case
#     if np.linalg.det(R) < 0:
#        Vt[m-1,:] *= -1
#        R = np.dot(Vt.T, U.T)

#     # translation
#     t = centroid_B.T - np.dot(R,centroid_A.T)

#     T = np.identity(m+1)
#     T[:m, :m] = R
#     T[:m, m] = t

#     return T


# def nearest_neighbor(src, dst):
#     '''
#     Find the nearest (Euclidean) neighbor in dst for each point in src
#     Input:
#         src: Nxm array of points
#         dst: Nxm array of points
#     Output:
#         distances: Euclidean distances of the nearest neighbor
#         indices: dst indices of the nearest neighbor
#     '''

#     neigh = NearestNeighbors(n_neighbors=1)
#     neigh.fit(dst)
#     distances, indices = neigh.kneighbors(src, return_distance=True)
#     return distances.ravel(), indices.ravel()


# def icp(source, target, init_transformation, max_iterations=100, tolerance=0.001):
#     '''
#     The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
#     Input:
#         A: Nxm numpy array of source mD points
#         B: Nxm numpy array of destination mD point
#         init_pose: (m+1)x(m+1) homogeneous transformation
#         max_iterations: exit algorithm after max_iterations
#         tolerance: convergence criteria
#     Output:
#         T: final homogeneous transformation that maps A on to B
#         distances: Euclidean distances (errors) of the nearest neighbor
#         i: number of iterations to converge
#     '''

#     # src=A
#     # dst=B

#     source.transform(init_transformation)

#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(source)

#     prev_error = 0
    
#     for i in range(max_iterations):
#         # find the nearest neighbors between the current source and destination points
#         distances, indices = nearest_neighbor(source_temp.points, target_temp.points)

#         # compute the transformation between the current source and nearest destination points
#         T= output_transform(A, B)

#         # update the current source
#         pcl_temp.transform(T)
#         A=pcl_temp.points

#         # check error
#         mean_error = np.mean(distances)
#         if np.abs(prev_error - mean_error) < tolerance:
#             break
#         prev_error = mean_error

#     # calculate final transformation
#     T= output_transform(source.transform(T), target)

#     return T
def euclidean_distance(point1, point2):
    """
    Euclidean distance between two points.
    :param point1: the first point as a tuple (a_1, a_2, ..., a_n)
    :param point2: the second point as a tuple (b_1, b_2, ..., b_n)
    :return: the Euclidean distance
    """
    a = np.array(point1)
    b = np.array(point2)

    return np.linalg.norm(a - b, ord=2)


def point_based_matching(point_pairs):
    """
    This function is based on the paper "Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans"
    by F. Lu and E. Milios.

    :param point_pairs: the matched point pairs [((x1, y1), (x1', y1')), ..., ((xi, yi), (xi', yi')), ...]
    :return: the rotation angle and the 2D translation (x, y) to be applied for matching the given pairs of points
    """

    x_mean = 0
    y_mean = 0
    xp_mean = 0
    yp_mean = 0
    n = len(point_pairs)

    if n == 0:
        return None, None, None

    for pair in point_pairs:

        (x, y), (xp, yp) = pair

        x_mean += x
        y_mean += y
        xp_mean += xp
        yp_mean += yp

    x_mean /= n
    y_mean /= n
    xp_mean /= n
    yp_mean /= n

    s_x_xp = 0
    s_y_yp = 0
    s_x_yp = 0
    s_y_xp = 0
    for pair in point_pairs:

        (x, y), (xp, yp) = pair

        s_x_xp += (x - x_mean)*(xp - xp_mean)
        s_y_yp += (y - y_mean)*(yp - yp_mean)
        s_x_yp += (x - x_mean)*(yp - yp_mean)
        s_y_xp += (y - y_mean)*(xp - xp_mean)

    rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
    translation_x = xp_mean - (x_mean*math.cos(rot_angle) - y_mean*math.sin(rot_angle))
    translation_y = yp_mean - (x_mean*math.sin(rot_angle) + y_mean*math.cos(rot_angle))

    return rot_angle, translation_x, translation_y


def icp(reference_points, points, max_iterations=100, distance_threshold=0.3, convergence_translation_threshold=1e-3,
        convergence_rotation_threshold=1e-4, point_pairs_threshold=10, verbose=False):
    """
    An implementation of the Iterative Closest Point algorithm that matches a set of M 2D points to another set
    of N 2D (reference) points.

    :param reference_points: the reference point set as a numpy array (N x 2)
    :param points: the point that should be aligned to the reference_points set as a numpy array (M x 2)
    :param max_iterations: the maximum number of iteration to be executed
    :param distance_threshold: the distance threshold between two points in order to be considered as a pair
    :param convergence_translation_threshold: the threshold for the translation parameters (x and y) for the
                                              transformation to be considered converged
    :param convergence_rotation_threshold: the threshold for the rotation angle (in rad) for the transformation
                                               to be considered converged
    :param point_pairs_threshold: the minimum number of point pairs the should exist
    :param verbose: whether to print informative messages about the process (default: False)
    :return: the transformation history as a list of numpy arrays containing the rotation (R) and translation (T)
             transformation in each iteration in the format [R | T] and the aligned points as a numpy array M x 2
    """

    transformation_history = []

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points)

    for iter_num in range(max_iterations):
        if verbose:
            print('------ iteration', iter_num, '------')

        closest_point_pairs = []  # list of point correspondences for closest point rule

        distances, indices = nbrs.kneighbors(points)
        for nn_index in range(len(distances)):
            if distances[nn_index][0] < distance_threshold:
                closest_point_pairs.append((points[nn_index], reference_points[indices[nn_index][0]]))

        # if only few point pairs, stop process
        if verbose:
            print('number of pairs found:', len(closest_point_pairs))
        if len(closest_point_pairs) < point_pairs_threshold:
            if verbose:
                print('No better solution can be found (very few point pairs)!')
            break

        # compute translation and rotation using point correspondences
        closest_rot_angle, closest_translation_x, closest_translation_y = point_based_matching(closest_point_pairs)
        if closest_rot_angle is not None:
            if verbose:
                print('Rotation:', math.degrees(closest_rot_angle), 'degrees')
                print('Translation:', closest_translation_x, closest_translation_y)
        if closest_rot_angle is None or closest_translation_x is None or closest_translation_y is None:
            if verbose:
                print('No better solution can be found!')
            break

        # transform 'points' (using the calculated rotation and translation)
        c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
        rot = np.array([[c, -s],
                        [s, c]])
        aligned_points = np.dot(points, rot.T)
        aligned_points[:, 0] += closest_translation_x
        aligned_points[:, 1] += closest_translation_y

        # update 'points' for the next iteration
        points = aligned_points

        # update transformation history
        transformation_history.append(np.hstack((rot, np.array([[closest_translation_x], [closest_translation_y]]))))

        # check convergence
        if (abs(closest_rot_angle) < convergence_rotation_threshold) \
                and (abs(closest_translation_x) < convergence_translation_threshold) \
                and (abs(closest_translation_y) < convergence_translation_threshold):
            if verbose:
                print('Converged!')
            break

    return transformation_history, points

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

            point.append([z*(j-256)/f,z*(i-256)/f,z])  
            color.append([col[0],col[1],col[2]])
            # if (z*(i-256)/f) >(-0.5):
            #     point.append([z*(j-256)/f,z*(i-256)/f,z])  
            #     color.append([col[0],col[1],col[2]])
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(point)
    pcl.colors = o3d.utility.Vector3dVector(color)
    # pcl.estimate_normals()

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