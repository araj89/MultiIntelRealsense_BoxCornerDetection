from pyntcloud.ransac.models import RansacPlane
import numpy as  np
import pandas as pd
import cv2
import math
from sklearn import linear_model

def get_best_fit_plane2(plane_pts):
    if plane_pts.shape[1] < 100:
        return [0, 0, 0, 0]
    pts = plane_pts.transpose()
    xy = pts[:, :2]
    z = pts[:, 2]
    ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=0.1)
    ransac.fit(xy, z)
    coeff = ransac.estimator_.coef_
    intercept = ransac.estimator_.intercept_

    return [coeff[0], coeff[1], -1, -intercept]


# get surface point and normal vector of the surface
# assume : points are in a surface
def get_best_fit_plane(plane_pts):
    if plane_pts.shape[1] < 100:
        return [0, 0, 0, 0]
    r_plane = RansacPlane(max_dist=0.0015)
    r_plane.least_squares_fit(plane_pts.transpose())
    #r_plane.fit(plane_pts.transpose())

    A = r_plane.normal[0]
    B = r_plane.normal[1]
    C = r_plane.normal[2]
    D = A * r_plane.point[0] + B * r_plane.point[1] + C * r_plane.point[2]

    if math.isnan(A) or math.isnan(B) or math.isnan(C):
        return [0, 0, 0, 0]
    return [A, B, C, D]

# get intersection point of 3 planes
def get_intersection(p1, p2, p3):
    #print ('---- plane equations ----')
    #print (p1)
    #print (p2)
    #print (p3)

    det_mat = np.array([[p1[0], p1[1], p1[2]],
                        [p2[0], p2[1], p2[2]],
                        [p3[0], p3[1], p3[2]]])
    det = np.linalg.det(det_mat)
    if det == 0:
        return np.array([0, 0, 0]).transpose()

    x_mat = np.array([[p1[3], p1[1], p1[2]],
                        [p2[3], p2[1], p2[2]],
                        [p3[3], p3[1], p3[2]]])
    x = np.linalg.det(x_mat) / det

    y_mat = np.array([[p1[0], p1[3], p1[2]],
                        [p2[0], p2[3], p2[2]],
                        [p3[0], p3[3], p3[2]]])
    y = np.linalg.det(y_mat) / det

    z_mat = np.array([[p1[0], p1[1], p1[3]],
                        [p2[0], p2[1], p2[3]],
                        [p3[0], p3[1], p3[3]]])
    z = np.linalg.det(z_mat) / det

    return np.array([x, y, z]).transpose()

# split point clouds to 3 surface points using corner point
def split_pt_clouds(pt_clouds, corner, corner_direction):
    if pt_clouds.shape[1] == 0:
        return None, None, None
    delta1 = 0.003
    #delta2 = 0.012 # 15mm
    #delta3 = 0.035 # 40mm
    delta2 = 0.003  # 15mm
    delta3 = 0.03  # 40mm

    b_xmin = corner_direction[0]
    b_ymin = corner_direction[1]

    xmin = np.min(pt_clouds[0, :])
    xmax = np.max(pt_clouds[0, :])
    ymin = np.min(pt_clouds[1, :])
    ymax = np.max(pt_clouds[1, :])
    zmin = np.min(pt_clouds[2, :])
    zmax = np.max(pt_clouds[2, :])

    z1 = corner[2] + delta1
    z2 = min(zmax, z1 + delta3)
    # extract x surface -------------------------------
    if b_ymin == 0:
        y1 = corner[1] + delta2
        y2 = min(ymax, y1 +delta3)
    else:
        y2 = corner[1] - delta2
        y1 = max(ymin, y2 - delta3)
    x_plane = pt_clouds[:, np.logical_and(pt_clouds[1, :] > y1, pt_clouds[1, :] < y2)]
    x_plane = x_plane[:, np.logical_and(x_plane[2, :] > z1, x_plane[2, :] < z2)]

    """xsub_min = np.min(x_plane[0, :])
    xsub_max = np.max(x_plane[0, :])
    xsub_med = np.median(x_plane[0, :])
    xsub_min = max(xsub_min, xsub_med - delta1)
    xsub_max = min(xsub_max, xsub_med + delta1)
    x_plane = x_plane[:, np.logical_and(x_plane[0, :] > xsub_min, x_plane[0, :] < xsub_max)]"""

    # extract y surface -------------------------------
    if b_xmin == 0:
        x1 = corner[0] + delta2
        x2 = min(xmax, x1 + delta3)
    else:
        x2 = corner[0] - delta2
        x1 = max(xmin, x2 - delta3)
    y_plane = pt_clouds[:, np.logical_and(pt_clouds[0, :] > x1, pt_clouds[0, :] < x2)]
    y_plane = y_plane[:, np.logical_and(y_plane[2, :] > z1, y_plane[2, :] < z2)]

    """ysub_min = np.min(y_plane[1, :])
    ysub_max = np.max(y_plane[1, :])
    ysub_med = np.median(y_plane[1, :])
    ysub_min = max(ysub_min, ysub_med - delta1)
    ysub_max = min(ysub_max, ysub_med + delta1)
    y_plane = y_plane[:, np.logical_and(y_plane[1, :] > ysub_min, y_plane[1, :] < ysub_max)]"""

    # extract z surface -------------------------------
    z_plane = pt_clouds[:,  np.logical_and(pt_clouds[0, :] > x1, pt_clouds[0, :] < x2)]
    z_plane = z_plane[:,  np.logical_and(z_plane[1, :] > y1, z_plane[1, :] < y2)]

    return x_plane, y_plane, z_plane

# get median x, y, z values of planes
def get_median_pos(x_plane, y_plane, z_plane, corner_direction):
    x_med = np.median(x_plane[0, :])
    x_min = np.min(x_plane[0, :])
    x_max = np.max(x_plane[0, :])
    x_delta = min(x_med - x_min, x_max - x_med)

    y_med = np.median(y_plane[1, :])
    y_min = np.min(y_plane[1, :])
    y_max = np.max(y_plane[1, :])
    y_delta = min(y_med - y_min, y_max - y_med)

    z_med = np.median(z_plane[2, :])
    z_min = np.min(z_plane[2, :])
    z_max = np.max(z_plane[2, :])
    z_delta = min(z_med - z_min, z_max - z_med)

    if corner_direction[0] == 0:
        x_med = x_max + x_delta
    else:
        x_med = x_min - x_delta

    if corner_direction[1] == 0:
        y_med = y_max + y_delta
    else:
        y_med = y_min - y_delta

    z_med = z_min - z_delta
    return np.array([x_med, y_med, z_med]).transpose()
# get intersection point from point cloud
def get_intersection_from_pointcloud(pt_clouds, corner, corner_direction):
    delta = 0.100 # 50mm

    x1 = corner[0] - delta
    x2 = corner[0] + delta
    y1 = corner[1] - delta
    y2 = corner[1] + delta
    z1 = corner[2] - delta
    z2 = corner[2] + delta

    corner_cloud = pt_clouds[:, np.logical_and(pt_clouds[0, :] > x1, pt_clouds[0, :] < x2)]
    corner_cloud = corner_cloud[:, np.logical_and(corner_cloud[1, :] > y1, corner_cloud[1, :] < y2)]
    corner_cloud = corner_cloud[:, np.logical_and(corner_cloud[2, :] > z1, corner_cloud[2, :] < z2)]

    x_plane, y_plane, z_plane = split_pt_clouds(corner_cloud, corner, corner_direction)
    if not np.any(x_plane):
        return [0, 0, 0]
    #print ('---- plane numbers ----')
    #print (x_plane.shape)
    #print (y_plane.shape)
    #print (z_plane.shape)

    #cumulatives = np.column_stack((x_plane, y_plane))
    #cumulatives = np.column_stack((cumulatives, z_plane))
    #return corner_cloud

    #return get_median_pos(x_plane, y_plane, z_plane, corner_direction)

    x_cof = get_best_fit_plane(x_plane)
    y_cof = get_best_fit_plane(y_plane)
    z_cof = get_best_fit_plane(z_plane)

    return get_intersection(x_cof, y_cof, z_cof)
    #return x_plane"""



