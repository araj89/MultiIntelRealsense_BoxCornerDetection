import pyrealsense2 as rs
import numpy as np
import cv2
from mypackages.realsense_device_manager import post_process_depth_frame
from mypackages.helper_functions import get_corner_3d, get_clipped_pointcloud, get_clipped_pointcloud2, \
    convert_depth_frame_to_pointcloud4, \
    convert_depth_frame_to_pointcloud2, convert_depth_frame_to_pointcloud5
import math
from mypackages.plane_intersection import get_intersection_from_pointcloud
# import pcl
from pylogix import PLC


def corners_from_pc(pcloud, corner_direction):
    n = pcloud.shape[1]
    if n == 0:
        print("----------------------------")
        return [-1, -1, -1]

    # simple clustering for x, y axis
    xmin = np.min(pcloud[0, :])
    xmax = np.max(pcloud[0, :])

    # ---------x scan----
    step = 0.003
    xscan = xmin
    while True:
        xscan += step
        if xscan >= xmax:
            break
        m = pcloud[:, np.logical_and(pcloud[0, :] >= xscan - step, pcloud[0, :] < xscan)]
        if m.shape[1] > 0:
            continue

        s = pcloud[:, pcloud[0, :] < xscan]
        l = pcloud[:, pcloud[0, :] >= xscan]

        if s.shape[1] > l.shape[1]:
            xmax = xscan
        else:
            xmin = xscan

    pcloud1 = pcloud[:, np.logical_and(pcloud[0, :] >= xmin, pcloud[0, :] <= xmax)]

    ymin = np.min(pcloud1[1, :])
    ymax = np.max(pcloud1[1, :])

    # ---------y scan------
    yscan = ymin
    while True:
        yscan += step
        if yscan >= ymax:
            break
        m = pcloud1[:, np.logical_and(pcloud1[1, :] >= yscan - step, pcloud1[1, :] < yscan)]
        if m.shape[1] > 0:
            continue

        s = pcloud1[:, pcloud1[1, :] < yscan]
        l = pcloud1[:, pcloud1[1, :] >= yscan]

        if s.shape[1] > l.shape[1]:
            ymax = yscan
        else:
            ymin = yscan

    pcloud2 = pcloud1[:, np.logical_and(pcloud1[1, :] >= ymin, pcloud1[1, :] <= ymax)]
    hmin = np.min(pcloud2[2, :])
    hmax = np.max(pcloud2[2, :])
    # print ('hmin, hmax : {}, {}'.format(hmin, hmax))

    if hmin < -0.2:
        h_delta = (hmax - hmin) / 5
        point_cloud = pcloud2[:, pcloud2[2, :] < ((hmin + hmax) / 2 + h_delta)]

        if point_cloud.shape[1] == 0:
            print('------- zero error -------')
            return [-1, -1, -1]

        xmin = np.min(point_cloud[0, :])
        xmax = np.max(point_cloud[0, :])
        ymin = np.min(point_cloud[1, :])
        ymax = np.max(point_cloud[1, :])

        b_xmin = corner_direction[0]
        b_ymin = corner_direction[1]

        if b_xmin == 0:
            x = xmin
        else:
            x = xmax
        if b_ymin == 0:
            y = ymin
        else:
            y = ymax

        min_error = 9999999
        id = -1
        for i in range(point_cloud.shape[1]):
            x0 = point_cloud[0][i]
            y0 = point_cloud[1][i]
            error = (x0 - x) * (x0 - x) + (y0 - y) * (y0 - y)
            if min_error > error:
                min_error = error
                id = i

        x = point_cloud[0][id]
        y = point_cloud[1][id]
        if b_xmin == 1:
            x += 0.002
        if b_xmin == 0:
            if b_ymin == 0: y-= 0.001
            else: y+= 0.001
        z = point_cloud[2][id]
        return np.array([x, y, z]).transpose()
    else:
        h_delta = (hmax - hmin) / 5
        point_cloud = pcloud2[:, pcloud2[2, :] < ((hmin + hmax) / 2 + h_delta)]

        if point_cloud.shape[1] == 0:
            print('------- zero error -------')
            return [-1, -1, -1]

        xmin = np.min(point_cloud[0, :])
        xmax = np.max(point_cloud[0, :])
        ymin = np.min(point_cloud[1, :])
        ymax = np.max(point_cloud[1, :])

        b_xmin = corner_direction[0]
        b_ymin = corner_direction[1]

        if b_xmin == 0:
            x = xmin
        else:
            x = xmax
        if b_ymin == 0:
            y = ymin
        else:
            y = ymax

        min_error = 9999999
        id = -1
        for i in range(point_cloud.shape[1]):
            x0 = point_cloud[0][i]
            y0 = point_cloud[1][i]
            error = (x0 - x) * (x0 - x) + (y0 - y) * (y0 - y)
            if min_error > error:
                min_error = error
                id = i

        if b_xmin == 1:
            x += 0.001
        if b_xmin == 1 and b_ymin == 1:
            y -= 0.002
        z = point_cloud[2][id]
        return np.array([x, y, z]).transpose()


def calculate_cumulative_pointcloud(frames_devices, calibration_info_devices, roi_2d, corner_direction,
                                    depth_threshold=0.01):
    """
 Calculate the cumulative pointcloud from the multiple devices
    Parameters:
    -----------
    frames_devices : dict
        The frames from the different devices
        keys: str
            Serial number of the device
        values: [frame]
            frame: rs.frame()
                The frameset obtained over the active pipeline from the realsense device

    calibration_info_devices : dict
        keys: str
            Serial number of the device
        values: [transformation_devices, intrinsics_devices]
            transformation_devices: Transformation object
                    The transformation object containing the transformation information between the device and the world coordinate systems
            intrinsics_devices: rs.intrinscs
                    The intrinsics of the depth_frame of the realsense device

    roi_2d : array
        The region of interest given in the following order [minX, maxX, minY, maxY]

    depth_threshold : double
        The threshold for the depth value (meters) in world-coordinates beyond which the point cloud information will not be used.
        Following the right-hand coordinate system, if the object is placed on the chessboard plane, the height of the object will increase along the negative Z-axis

    Return:
    ----------
    point_cloud_cumulative : array
        The cumulative pointcloud from the multiple devices
    """
    # Use a threshold of 5 centimeters from the chessboard as the area where useful points are found

    point_cloud_cumulative = np.array([-1, -1, -1]).transpose()
    for (device, frame) in frames_devices.items():
        # Filter the depth_frame using the Temporal filter and get the corresponding pointcloud for each frame
        filtered_depth_frame = post_process_depth_frame(frame[rs.stream.depth], temporal_smooth_alpha=0.1,
                                                        temporal_smooth_delta=40)
        depth_img = np.asarray(filtered_depth_frame.get_data())
        cv2.imwrite("depth_{}.png".format(device), depth_img)

        point_cloud, ptc2 = convert_depth_frame_to_pointcloud5(depth_img.copy(),
                                                         calibration_info_devices[device][1][rs.stream.depth])
        # corner_3d = get_corner_3d(np.asarray(filtered_depth_frame.get_data()), calibration_info_devices[device][1][rs.stream.depth])
        point_cloud = np.asanyarray(point_cloud)
        ptc2 = np.asanyarray(ptc2)

        # Get the point cloud in the world-coordinates using the transformation
        point_cloud = calibration_info_devices[device][0].apply_transformation(point_cloud)
        ptc2 = calibration_info_devices[device][0].apply_transformation(ptc2)

        # Filter the point cloud based on the depth of the object
        # The object placed has its height in the negative direction of z-axis due to the right-hand coordinate system
        # point_cloud = get_clipped_pointcloud(point_cloud, roi_2d)
        point_cloud = get_clipped_pointcloud2(point_cloud, roi_2d[0], roi_2d[2], roi_2d[1], roi_2d[3])
        point_cloud = point_cloud[:, point_cloud[2, :] < -depth_threshold]
        point_cloud = point_cloud.astype(np.float32)

        ptc2 = get_clipped_pointcloud2(ptc2, roi_2d[0], roi_2d[2], roi_2d[1], roi_2d[3])
        ptc2 = ptc2[:, ptc2[2, :] < -depth_threshold]
        ptc2 = ptc2.astype(np.float32)

        # -------------------------   get corner ----------------------
        corner2 = corners_from_pc(point_cloud, corner_direction[device])
        corner = get_intersection_from_pointcloud(ptc2, corner2, corner_direction[device])
        # -------------------------------------------------------------
        # if corner[0] != -1 and corner[1] != -1 and corner[2] != -1:
        point_cloud_cumulative = np.column_stack((point_cloud_cumulative, corner))
    point_cloud_cumulative = np.delete(point_cloud_cumulative, 0, 1)

    #print (point_cloud_cumulative.shape)

    return point_cloud_cumulative


def calculate_boundingbox_points(point_cloud, calibration_info_devices, depth_threshold=0.01):
    """
    Calculate the top and bottom bounding box corner points for the point cloud in the image coordinates of the color imager of the realsense device

    Parameters:
    -----------
    point_cloud : ndarray
        The (3 x N) array containing the pointcloud information

    calibration_info_devices : dict
        keys: str
            Serial number of the device
        values: [transformation_devices, intrinsics_devices, extrinsics_devices]
            transformation_devices: Transformation object
                    The transformation object containing the transformation information between the device and the world coordinate systems
            intrinsics_devices: rs.intrinscs
                    The intrinsics of the depth_frame of the realsense device
            extrinsics_devices: rs.extrinsics
                    The extrinsics between the depth imager 1 and the color imager of the realsense device

    depth_threshold : double
        The threshold for the depth value (meters) in world-coordinates beyond which the point cloud information will not be used
        Following the right-hand coordinate system, if the object is placed on the chessboard plane, the height of the object will increase along the negative Z-axis

    Return:
    ----------
    bounding_box_points_color_image : dict
        The bounding box corner points in the image coordinate system for the color imager
        keys: str
                Serial number of the device
            values: [points]
                points: list
                    The (8x2) list of the upper corner points stacked above the lower corner points

    length : double
        The length of the bounding box calculated in the world coordinates of the pointcloud

    width : double
        The width of the bounding box calculated in the world coordinates of the pointcloud

    height : double
        The height of the bounding box calculated in the world coordinates of the pointcloud
    """
    # Calculate the dimensions of the filtered and summed up point cloud
    # Some dirty array manipulations are gonna follow
    if point_cloud.shape[1] >= 4:
        w0 = math.sqrt((point_cloud[0][0] - point_cloud[0][1]) * (point_cloud[0][0] - point_cloud[0][1]) + (
                point_cloud[1][0] - point_cloud[1][1]) * (point_cloud[1][0] - point_cloud[1][1]) + (
                               point_cloud[2][0] - point_cloud[2][1]) * (point_cloud[2][0] - point_cloud[2][1]))
        w1 = math.sqrt((point_cloud[0][2] - point_cloud[0][3]) * (point_cloud[0][2] - point_cloud[0][3]) + (
                point_cloud[1][2] - point_cloud[1][3]) * (point_cloud[1][2] - point_cloud[1][3]) + (
                               point_cloud[2][2] - point_cloud[2][3]) * (point_cloud[2][2] - point_cloud[2][3]))
        w = int((w0 + w1) / 2 * 1000 + 0.5)

        h0 = math.sqrt((point_cloud[0][2] - point_cloud[0][1]) * (point_cloud[0][2] - point_cloud[0][1]) + (
                point_cloud[1][2] - point_cloud[1][1]) * (point_cloud[1][2] - point_cloud[1][1]) + (
                               point_cloud[2][2] - point_cloud[2][1]) * (point_cloud[2][2] - point_cloud[2][1]))
        h1 = math.sqrt((point_cloud[0][0] - point_cloud[0][3]) * (point_cloud[0][0] - point_cloud[0][3]) + (
                point_cloud[1][0] - point_cloud[1][3]) * (point_cloud[1][0] - point_cloud[1][3]) + (
                               point_cloud[2][0] - point_cloud[2][3]) * (point_cloud[2][0] - point_cloud[2][3]))
        h = int((h0 + h1) / 2 * 1000 + 0.5)

        bounding_box_world_3d = point_cloud

        bounding_box_points_color_image = {}
        bounding_box_points_infrared_image = {}
        for (device, calibration_info) in calibration_info_devices.items():
            # Transform the bounding box corner points to the device coordinates
            bounding_box_device_3d = calibration_info[0].inverse().apply_transformation(bounding_box_world_3d)

            # Obtain the image coordinates in the color imager using the bounding box 3D corner points in the device coordinates
            color_pixel = []
            infrared_pixel = []
            bounding_box_device_3d = bounding_box_device_3d.transpose().tolist()
            for bounding_box_point in bounding_box_device_3d:
                bounding_box_color_image_point = rs.rs2_transform_point_to_point(calibration_info[2],
                                                                                 bounding_box_point)
                color_pixel.append(
                    rs.rs2_project_point_to_pixel(calibration_info[1][rs.stream.color], bounding_box_color_image_point))
                infrared_pixel.append(
                    rs.rs2_project_point_to_pixel(calibration_info[1][(rs.stream.infrared, 1)], bounding_box_point))
            bounding_box_points_color_image[device] = np.row_stack(color_pixel)
            bounding_box_points_infrared_image[device] = np.row_stack(infrared_pixel)

        return bounding_box_points_color_image, bounding_box_points_infrared_image, w, h
    else:
        return {}, {}, 0, 0


def visualise_measurements_infrared(frames_devices, bounding_box_points_devices, length, width):
    """
 Calculate the cumulative pointcloud from the multiple devices

    Parameters:
    -----------
    frames_devices : dict
        The frames from the different devices
        keys: str
            Serial number of the device
        values: [frame]
            frame: rs.frame()
                The frameset obtained over the active pipeline from the realsense device

    bounding_box_points_color_image : dict
        The bounding box corner points in the image coordinate system for the color imager
        keys: str
                Serial number of the device
            values: [points]
                points: list
                    The (8x2) list of the upper corner points stacked above the lower corner points

    length : double
        The length of the bounding box calculated in the world coordinates of the pointcloud

    width : double
        The width of the bounding box calculated in the world coordinates of the pointcloud

    height : double
        The height of the bounding box calculated in the world coordinates of the pointcloud
    """
    for (device, frame) in frames_devices.items():
        if length != 0 and width != 0:
            corners = bounding_box_points_devices[device]
        infrared_image = np.asarray(frame[(rs.stream.infrared, 1)].get_data())
        infrared_image = cv2.cvtColor(infrared_image, cv2.COLOR_GRAY2BGR)
        if (length != 0 and width != 0):
            box_info = "Length, Width: " + str(int(length * 1000)) + ", " + str(
                int(width * 1000))

            cv2.putText(infrared_image, box_info, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
            print(corners.shape)
            for i in range(len(corners)):
                # print (corners[i])
                cv2.circle(infrared_image, (int(corners[i][0]), int(corners[i][1])), 5, (0, 0, 255))

        # Visualise the results
        cv2.imshow('Color image from RealSense Device Nr: ' + device, infrared_image)
        cv2.waitKey(1)


def visualise_measurements_color(frames_devices, bounding_box_points_devices, corners_3d, length, width):
    """
 Calculate the cumulative pointcloud from the multiple devices

    Parameters:
    -----------
    frames_devices : dict
        The frames from the different devices
        keys: str
            Serial number of the device
        values: [frame]
            frame: rs.frame()
                The frameset obtained over the active pipeline from the realsense device

    bounding_box_points_color_image : dict
        The bounding box corner points in the image coordinate system for the color imager
        keys: str
                Serial number of the device
            values: [points]
                points: list
                    The (8x2) list of the upper corner points stacked above the lower corner points

    length : double
        The length of the bounding box calculated in the world coordinates of the pointcloud

    width : double
        The width of the bounding box calculated in the world coordinates of the pointcloud

    height : double
        The height of the bounding box calculated in the world coordinates of the pointcloud
    """

    for (device, frame) in frames_devices.items():
        if length != 0 and width != 0:
            corners = bounding_box_points_devices[device]
        infrared_image = np.asarray(frame[rs.stream.color].get_data())
        h, w, _ = infrared_image.shape

        # infrared_image = cv2.cvtColor(infrared_image, cv2.COLOR_GRAY2BGR)
        if (length != 0 and width != 0):
            box_info = "Length, Width: " + str(length) + ", " + str(width)

            cv2.putText(infrared_image, box_info, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
            # print (corners.shape)
            for i in range(len(corners)):
                x = int(corners[i][0] + 0.5)
                y = int(corners[i][1] + 0.5)
                cv2.circle(infrared_image, (x, y), 5, (0, 0, 255))

                cx = corners_3d[0][i] * 1000
                cy = corners_3d[1][i] * 1000
                cz = -corners_3d[2][i] * 1000
                cx = round(cx, 3)
                cy = round(cy, 3)
                cz = round(cz, 3)
                pt_text = "({}, {}, {})".format(cx, cy, cz)
                print(pt_text)

                # ---------------------DAC Additions-----------------------
                # with open("Output.txt", "a") as text_file:
                #	print (pt_text, file=text_file)

                # Send Coords to PLC Arrays
                #with PLC() as comm:
                #    comm.IPAddress = '10.5.6.190'
                #    comm.Write('pyi_CoordX[' + str(i) + ']', cx)
                #    comm.Write('pyi_CoordY[' + str(i) + ']', cy)
                #    comm.Write('pyi_CoordZ[' + str(i) + ']', cz)
                # Update flag to PLC when 4th coordinate has been written
                # if i >= 3:
                #	comm.Write('pyi_CoordNew', 1)

                # comm.Close()
                # ---------------------END DAC Additions-----------------------

                if x < 400:
                    px = x
                    py = y + 50
                elif y > 900:
                    px = x - 200
                    py = y + 50
                else:
                    px = x
                    py = y + 50
                cv2.putText(infrared_image, pt_text, (px, py), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255))
            print('---------')

            # ---------------------DAC Additions-----------------------
            # Update PLC flag when all coords have been written
            #with PLC() as comm:
            #    comm.IPAddress = '10.5.6.190'
            #    comm.Write('pyi_CoordNew', 1)
            #    comm.Close()
        # ---------------------END DAC Additions-----------------------

        # Visualise the results
        cv2.imshow('Color image from RealSense Device Nr: ' + device, infrared_image)
        cv2.waitKey(1)