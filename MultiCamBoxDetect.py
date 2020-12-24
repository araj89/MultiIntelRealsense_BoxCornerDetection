
# Import RealSense, OpenCV and NumPy
import pyrealsense2 as rs
import cv2
import numpy as np
from collections import defaultdict
from mypackages.realsense_device_manager import DeviceManager, post_process_depth_frame
from mypackages.helper_functions import get_boundary_corners_2D, convert_depth_pixel_to_metric_coordinate, \
    cv_find_chessboard, get_chessboard_points_3D, get_depth_at_pixel, convert_depth_pixel_to_metric_coordinate, cv_find_chessboard2
from mypackages.measurement_task2 import calculate_boundingbox_points, calculate_cumulative_pointcloud, visualise_measurements_infrared, visualise_measurements_color
import pickle
import mypackages.calculate_rmsd_kabsch as rmsd


B_CALIBED = True
SHOW_DEPTH = False
depth_thres = 0.05

# --------- internal paramters-------

# Define some constants
resolution_width = 1280  # pixels
resolution_height = 720  # pixels
frame_rate = 6 # fps
dispose_frames_for_stablisation = 30  # frames

chessboard_params = {
    '002422061290' : [8, 9, 0.065],
    '002422062653' : [8, 11, 0.065],
    '020122061887' : [8, 7, 0.065],
    '020122063035' : [8, 11, 0.065]
}

chessboard_origins = {
    '002422061290' : [0, 0],
    '002422062653' : [0, 0],
    '020122061887' : [0, 0],
    '020122063035' : [0, 0]
}

chessboard_corner_direction = {
    '002422061290' : [0, 1],
    '002422062653' : [0, 0],
    '020122061887' : [1, 1],
    '020122063035' : [1, 0]
}
roi_2d = [-0.1, -0.1, 0.535, 0.65] # [sx, sy, ex, ey]
calib_file = 'mypackages/calib_data.json'

def calculate_transformation_kabsch(src_points, dst_points):
    assert src_points.shape == dst_points.shape
    if src_points.shape[0] != 3:
        raise Exception("The input data matrix had to be transposed in order to compute transformation.")

    src_points = src_points.transpose()
    dst_points = dst_points.transpose()

    src_points_centered = src_points - rmsd.centroid(src_points)
    dst_points_centered = dst_points - rmsd.centroid(dst_points)

    rotation_matrix = rmsd.kabsch(src_points_centered, dst_points_centered)
    rmsd_value = rmsd.kabsch_rmsd(src_points_centered, dst_points_centered)

    translation_vector = rmsd.centroid(dst_points) - np.matmul(rmsd.centroid(src_points), rotation_matrix)

    return rotation_matrix.transpose(), translation_vector.transpose(), rmsd_value

class Transformation:
    def __init__(self, rotation_matrix, translation_vector):
        self.pose_mat = np.zeros((4, 4))
        self.pose_mat[:3, :3] = rotation_matrix
        self.pose_mat[:3, 3] = translation_vector.flatten()
        self.pose_mat[3, 3] = 1

    def apply_transformation(self, points):
        assert (points.shape[0] == 3)
        n = points.shape[1]
        points_ = np.vstack((points, np.ones((1, n))))
        points_trans_ = np.matmul(self.pose_mat, points_)
        points_transformed = np.true_divide(points_trans_[:3, :], points_trans_[[-1], :])
        return points_transformed

    def inverse(self):
        rotation_matrix = self.pose_mat[:3, :3]
        translation_vector = self.pose_mat[:3, 3]

        rot = np.transpose(rotation_matrix)
        trans = - np.matmul(np.transpose(rotation_matrix), translation_vector)
        return Transformation(rot, trans)

class PoseEstimation:
    def __init__(self, frames, intrinsic, chessboard_params):
        self.frames = frames
        self.intrinsic = intrinsic
        self.chessboard_params = chessboard_params

    def get_chessboard_corners_in3d(self):
        corners3D = {}
        for (serial, frameset) in self.frames.items():
            depth_frame = post_process_depth_frame(frameset[rs.stream.depth])
            infrared_frame = frameset[(rs.stream.infrared, 1)]
            depth_intrinsics = self.intrinsic[serial][rs.stream.depth]

            # remove remain chessboard area---------------------
            infrared_img = np.asanyarray(infrared_frame.get_data()).copy()
            if serial == '002422061290':
                infrared_img = cv2.imread('002422061290.png', cv2.IMREAD_GRAYSCALE)
                cnt = np.array([[700, 712], [1165, 541], [1279, 719]], dtype=np.int32)
                cv2.drawContours(infrared_img, [cnt], -1, 255, cv2.FILLED)
            elif serial == '020122061887':
                infrared_img = cv2.imread('020122061887.png', cv2.IMREAD_GRAYSCALE)
                cnt = np.array([[450, 410], [866, 665], [352, 713]], dtype=np.int32)
                cv2.drawContours(infrared_img, [cnt], -1, 255, cv2.FILLED)
            elif serial == '002422062653':
                infrared_img = cv2.imread('002422062653.png', cv2.IMREAD_GRAYSCALE)
            elif serial == '020122063035':
                infrared_img = cv2.imread('020122063035.png', cv2.IMREAD_GRAYSCALE)

            # ------------------------

            #found_corners, points2D = cv_find_chessboard(depth_frame, infrared_frame, self.chessboard_params[serial])
            found_corners, points2D = cv_find_chessboard2(infrared_img, self.chessboard_params[serial])
            corners3D[serial] = [found_corners, None, None, None]
            if found_corners:
                points3D = np.zeros((3, len(points2D[0])))
                validPoints = [False] * len(points2D[0])
                for index in range(len(points2D[0])):
                    corner = points2D[:, index].flatten()
                    depth = get_depth_at_pixel(depth_frame, corner[0], corner[1])
                    if depth != 0 and depth is not None:
                        validPoints[index] = True
                        [X, Y, Z] = convert_depth_pixel_to_metric_coordinate(depth, corner[0], corner[1],
                                                                             depth_intrinsics)
                        points3D[0, index] = X
                        points3D[1, index] = Y
                        points3D[2, index] = Z
                corners3D[serial] = found_corners, points2D, points3D, validPoints
        return corners3D

    def perform_pose_estimation(self):
        corners3D = self.get_chessboard_corners_in3d()
        retval = {}
        for (serial, [found_corners, points2D, points3D, validPoints]) in corners3D.items():
            objectpoints = get_chessboard_points_3D(self.chessboard_params[serial])
            retval[serial] = [False, None, None, None]
            if found_corners == True:
                # initial vectors are just for correct dimension
                valid_object_points = objectpoints[:, validPoints]
                valid_observed_object_points = points3D[:, validPoints]
                print('number of valid corners {}'.format(valid_object_points.shape[1]))
                # check for sufficient points
                if valid_object_points.shape[1] < 5:
                    print("Not enough points have a valid depth for calculating the transformation")

                else:
                    [rotation_matrix, translation_vector, rmsd_value] = calculate_transformation_kabsch(
                        valid_object_points, valid_observed_object_points)
                    retval[serial] = [True, Transformation(rotation_matrix, translation_vector), points2D, rmsd_value]
                    print("RMS error for calibration with device number", serial, "is :", rmsd_value, "m")
        return retval

    def find_chessboard_boundary_for_depth_image(self):
        boundary = {}

        for (serial, frameset) in self.frames.items():
            depth_frame = post_process_depth_frame(frameset[rs.stream.depth])
            infrared_frame = frameset[(rs.stream.infrared, 1)]
            found_corners, points2D = cv_find_chessboard(depth_frame, infrared_frame, self.chessboard_params[serial])
            boundary[serial] = [np.floor(np.amin(points2D[0, :])).astype(int),
                                np.floor(np.amax(points2D[0, :])).astype(int),
                                np.floor(np.amin(points2D[1, :])).astype(int),
                                np.floor(np.amax(points2D[1, :])).astype(int)]

        return boundary

def visualize_depth(depth_frame, min_shift = 0, max_shift = 0):
    depth_image = post_process_depth_frame(depth_frame, temporal_smooth_alpha=0.5, temporal_smooth_delta=100)
    depth_image = np.asarray(depth_image.get_data())

    _min, _max, _, _ = cv2.minMaxLoc(depth_image)

    cut_max = max(_min + 255, _max - max_shift)
    cut_min = min(_min + min_shift, cut_max - 255)
    depth_image = np.where((depth_image > cut_max), cut_max, depth_image)
    depth_image = np.where((depth_image < cut_min), cut_min, depth_image)

    delta = (cut_max - cut_min) / 255
    depth_image = depth_image - cut_min
    depth_image = depth_image / delta
    depth_image = depth_image.astype(np.uint8)
    #depth_image = cv2.equalizeHist(depth_image)
    return depth_image

def get_devices():
    # Enable the streams from all the intel realsense devices
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
    rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
    rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)
    # Use the device manager class to enable the devices and get the frames
    device_manager = DeviceManager(rs.context(), rs_config, exposure=100000, laser_power=270)
    device_manager.enable_all_devices()

    return device_manager

def calibration(device_manager, intrinsics_devices):
    # Estimate the pose of the chessboard in the world coordinate using the Kabsch Method
    calibrated_device_count = 0
    while calibrated_device_count < len(device_manager._available_devices):
        frames = device_manager.poll_frames()
        pose_estimator = PoseEstimation(frames, intrinsics_devices, chessboard_params)
        transformation_result_kabsch = pose_estimator.perform_pose_estimation()
        calibrated_device_count = 0
        for device in device_manager._available_devices:
            if not transformation_result_kabsch[device][0]:
                print("Place the chessboard on the plane where the object needs to be detected: {}".format(device))
            else:
                calibrated_device_count += 1
    cv2.destroyAllWindows()
    print("Calibration completed... \nPlace the box in the field of view of the devices...")
    return transformation_result_kabsch

def get_corners():
    device_manager = get_devices()
    try:
        # Allow some frames for the auto-exposure controller to stablise
        cnt = 0
        for frame in range(dispose_frames_for_stablisation):
            frames = device_manager.poll_frames()
            cnt += 1
            print ('dispose frame : {}'.format(cnt))

        assert (len(device_manager._available_devices) > 0)

        # Get the intrinsics of the realsense device
        intrinsics_devices = device_manager.get_device_intrinsics(frames)
        print ('get intrinsic paramter')
        if not B_CALIBED:
            calibed_data = calibration(device_manager, intrinsics_devices)
            with open(calib_file, 'wb') as f:
                pickle.dump(calibed_data, f)
        else:
            f = open(calib_file, 'rb')
            calibed_data = pickle.load(f)

        transformation_result_kabsch = calibed_data

        transformation_devices = {}

        for device in device_manager._available_devices:
            transformation_devices[device] = transformation_result_kabsch[device][1].inverse()

        extrinsics_devices = device_manager.get_depth_to_color_extrinsics(frames)

        # Get the calibration info as a dictionary to help with display of the measurements onto the color image instead of infra red image
        calibration_info_devices = defaultdict(list)
        for calibration_info in (transformation_devices, intrinsics_devices, extrinsics_devices):
            for key, value in calibration_info.items():
                calibration_info_devices[key].append(value)

        # Enable the emitter of the devices
        device_manager.enable_emitter(True)
        device_manager.enable_auto_exposure()
        # Load the JSON settings file in order to enable High Accuracy preset for the realsense
        device_manager.load_settings_json("mypackages/GoodState.json")

        # Continue acquisition until terminated with Ctrl+C by the user
        while 1:
            # Get the frames from all the devices
            frames_devices = device_manager.poll_frames()
            if SHOW_DEPTH:
                for d, frameset in frames_devices.items():
                    depth_image = visualize_depth(frameset[rs.stream.depth])
                    cv2.imshow(d, depth_image)
                    cv2.waitKey(1)

            # Calculate the pointcloud using the depth frames from all the devices
            point_cloud = calculate_cumulative_pointcloud(frames_devices, calibration_info_devices, roi_2d, chessboard_corner_direction, depth_thres)

            # Get the bounding box for the pointcloud in image coordinates of the color imager
            bounding_box_points_color_image, bounding_box_points_infrared_image, length, width= calculate_boundingbox_points(point_cloud,
                                                                                                  calibration_info_devices, depth_thres)

            #print ('edge points :{}'.format(interpt_3d))
            # ----------------------------------
            # Draw the bounding box points on the color image and visualise the results
            #visualise_measurements(frames_devices, bounding_box_points_color_image, length, width, height)
            #visualise_measurements_infrared(frames_devices, bounding_box_points_infrared_image, length, width)
            visualise_measurements_color(frames_devices, bounding_box_points_color_image, point_cloud, length, width)

    except KeyboardInterrupt:
        print("The program was interupted by the user. Closing the program...")

    finally:
        device_manager.disable_streams()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    get_corners()
