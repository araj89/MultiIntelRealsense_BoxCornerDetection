##################################################################################################
##       License: Apache 2.0. See LICENSE file in root directory.		                      ####
##################################################################################################
##                  Box Dimensioner with multiple cameras: Helper files 					  ####
##################################################################################################

# Opencv helper functions and class
import cv2
import numpy as np
import imutils

"""
  _   _        _                      _____                     _    _
 | | | |  ___ | | _ __    ___  _ __  |  ___|_   _  _ __    ___ | |_ (_)  ___   _ __   ___
 | |_| | / _ \| || '_ \  / _ \| '__| | |_  | | | || '_ \  / __|| __|| | / _ \ | '_ \ / __|
 |  _  ||  __/| || |_) ||  __/| |    |  _| | |_| || | | || (__ | |_ | || (_) || | | |\__ \
 |_| |_| \___||_|| .__/  \___||_|    |_|    \__,_||_| |_| \___| \__||_| \___/ |_| |_||___/
				 _|
"""


def calculate_rmsd(points1, points2, validPoints=None):
    """
	calculates the root mean square deviation between to point sets

	Parameters:
	-------
	points1, points2: numpy matrix (K, N)
	where K is the dimension of the points and N is the number of points

	validPoints: bool sequence of valid points in the point set.
	If it is left out, all points are considered valid
	"""
    assert (points1.shape == points2.shape)
    N = points1.shape[1]

    if validPoints == None:
        validPoints = [True] * N

    assert (len(validPoints) == N)

    points1 = points1[:, validPoints]
    points2 = points2[:, validPoints]

    N = points1.shape[1]

    dist = points1 - points2
    rmsd = 0
    for col in range(N):
        rmsd += np.matmul(dist[:, col].transpose(), dist[:, col]).flatten()[0]

    return np.sqrt(rmsd / N)


def get_chessboard_points_3D(chessboard_params):
    """
	Returns the 3d coordinates of the chessboard corners
	in the coordinate system of the chessboard itself.

	Returns
	-------
	objp : array
		(3, N) matrix with N being the number of corners
	"""
    assert (len(chessboard_params) == 3)
    width = chessboard_params[0]
    height = chessboard_params[1]
    square_size = chessboard_params[2]
    objp = np.zeros((width * height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    return objp.transpose() * square_size


def cv_find_chessboard(depth_frame, infrared_frame, chessboard_params):
    """
	Searches the chessboard corners using the set infrared image and the
	checkerboard size

	Returns:
	-----------
	chessboard_found : bool
						  Indicates wheather the operation was successful
	corners          : array
						  (2,N) matrix with the image coordinates of the chessboard corners
	"""
    assert (len(chessboard_params) == 3)
    print (chessboard_params)
    infrared_image = np.asanyarray(infrared_frame.get_data())
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_found = False
    chessboard_found, corners = cv2.findChessboardCorners(infrared_image, (
        chessboard_params[0], chessboard_params[1]), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS)

    if chessboard_found:
        for i in range(corners.shape[0]):
            v = 255
            if i == 0:
                v = 0
            cv2.circle(infrared_image, (corners[i][0][0], corners[i][0][1]), 3, v, 3)
        cv2.imshow('chessboard corners', infrared_image)
        cv2.waitKey(0)
        corners = cv2.cornerSubPix(infrared_image, corners, (11, 11), (-1, -1), criteria)
        corners = np.transpose(corners, (2, 0, 1))
    return chessboard_found, corners

def cv_find_chessboard2(infrared_img, chessboard_params):
    """
	Searches the chessboard corners using the set infrared image and the
	checkerboard size

	Returns:
	-----------
	chessboard_found : bool
						  Indicates wheather the operation was successful
	corners          : array
						  (2,N) matrix with the image coordinates of the chessboard corners
	"""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_found = False
    chessboard_found, corners = cv2.findChessboardCorners(infrared_img, (
        chessboard_params[0], chessboard_params[1]), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS)

    infrared_img2 = infrared_img.copy()
    if chessboard_found:
        for i in range(corners.shape[0]):
            v = 255
            if i == 0:
                v = 0
            cv2.circle(infrared_img2, (corners[i][0][0], corners[i][0][1]), 3, v, 3)

        corners = cv2.cornerSubPix(infrared_img, corners, (11, 11), (-1, -1), criteria)
        corners = np.transpose(corners, (2, 0, 1))
    cv2.imshow('chessboard corners', infrared_img2)
    cv2.waitKey(0)

    return chessboard_found, corners


def get_depth_at_pixel(depth_frame, pixel_x, pixel_y):
    """
	Get the depth value at the desired image point

	Parameters:
	-----------
	depth_frame 	 : rs.frame()
						   The depth frame containing the depth information of the image coordinate
	pixel_x 	  	 	 : double
						   The x value of the image coordinate
	pixel_y 	  	 	 : double
							The y value of the image coordinate

	Return:
	----------
	depth value at the desired pixel

	"""
    return depth_frame.as_depth_frame().get_distance(round(pixel_x), round(pixel_y))


def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y, camera_intrinsics):
    """
	Convert the depth and image point information to metric coordinates

	Parameters:
	-----------
	depth 	 	 	 : double
						   The depth value of the image point
	pixel_x 	  	 	 : double
						   The x value of the image coordinate
	pixel_y 	  	 	 : double
							The y value of the image coordinate
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed

	Return:
	----------
	X : double
		The x value in meters
	Y : double
		The y value in meters
	Z : double
		The z value in meters

	"""
    X = (pixel_x - camera_intrinsics.ppx) / camera_intrinsics.fx * depth
    Y = (pixel_y - camera_intrinsics.ppy) / camera_intrinsics.fy * depth
    return X, Y, depth


def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics):
    """
	Convert the depthmap to a 3D point cloud

	Parameters:
	-----------
	depth_frame 	 	 : rs.frame()
						   The depth_frame containing the depth map
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed

	Return:
	----------
	x : array
		The x values of the pointcloud in meters
	y : array
		The y values of the pointcloud in meters
	z : array
		The z values of the pointcloud in meters

	"""

    [height, width] = depth_image.shape

    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx) / camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy) / camera_intrinsics.fy

    # stablize depth values
    kernel = np.ones((1, 1), np.uint8)
    kernel2 = np.ones((1, 1), np.uint8)

    _, thres_img = cv2.threshold(depth_image, 1, 255, cv2.THRESH_BINARY)
    thres_img = cv2.erode(thres_img, kernel, None, iterations=1)

    depth_image = cv2.dilate(depth_image, kernel2, None, iterations=1)
    # depth_image = cv2.erode(depth_image, kernel2, None, iterations=1)

    b = thres_img.flatten()
    z = depth_image.flatten() / 1000
    x = np.multiply(x, z)
    y = np.multiply(y, z)

    x = x[np.nonzero(b)]
    y = y[np.nonzero(b)]
    z = z[np.nonzero(b)]

    return x, y, z

def convert_depth_frame_to_pointcloud2(depth_image, camera_intrinsics):
    """
	Convert the depthmap to a 3D point cloud

	Parameters:
	-----------
	depth_frame 	 	 : rs.frame()
						   The depth_frame containing the depth map
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed

	Return:
	----------
	x : array
		The x values of the pointcloud in meters
	y : array
		The y values of the pointcloud in meters
	z : array
		The z values of the pointcloud in meters

	"""

    [height, width] = depth_image.shape

    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx) / camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy) / camera_intrinsics.fy

    kernel1 = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((11, 11), np.uint8)
    kernel3 = np.ones((23, 23), np.uint8)

    depth_image = np.where((depth_image == 0), 5000, depth_image)
    img1 = cv2.erode(depth_image, kernel1, None, iterations=1)
    img2 = cv2.erode(depth_image, kernel2, None, iterations=1)
    img3 = cv2.erode(depth_image, kernel3, None, iterations=1)

    res1 = np.where((img1 == img2), 255, 0)
    res2 = np.where((img1 == img3), 255, 0)
    res3 = np.where((img1 == depth_image), 255, 0)

    res1 = np.where((res1 == res2), res1, 0)
    res2 = np.where((res2 == res3), res2, 0)
    res1 = np.where((res1 == res2), res1, 0)
    res1 = res1.astype(np.uint8)

    """res2 = np.zeros((height, width), np.uint8)
    cnts = cv2.findContours(res1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for cnt in cnts:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        res2[cy, cx] = 255"""

    depth_image = np.where((res1 == 255), depth_image, 0)
    depth_image = np.where((depth_image == 5000), 0, depth_image)
    z = depth_image.flatten() / 1000



    x = np.multiply(x, z)
    y = np.multiply(y, z)

    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]

    return x, y, z

def convert_depth_frame_to_pointcloud3(depth_image, camera_intrinsics):
    """
	Convert the depthmap to a 3D point cloud

	Parameters:
	-----------
	depth_frame 	 	 : rs.frame()
						   The depth_frame containing the depth map
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed

	Return:
	----------
	x : array
		The x values of the pointcloud in meters
	y : array
		The y values of the pointcloud in meters
	z : array
		The z values of the pointcloud in meters

	"""

    [height, width] = depth_image.shape

    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx) / camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy) / camera_intrinsics.fy

    kernel1 = np.array([[0, 0, 1, 0, 0],
                        [0, 1, 0, 1, 0],
                        [1, 0, 0, 0, 1],
                        [0, 1, 0, 1, 0],
                        [0, 0, 1, 0, 0]], dtype=np.uint8)
    kernel2 = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=np.uint8)
    kernel3 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    depth_image = np.where((depth_image == 0), 5000, depth_image)
    img1 = cv2.erode(depth_image, kernel1, None, iterations=1)
    img2 = cv2.erode(depth_image, kernel2, None, iterations=1)
    img3 = cv2.erode(depth_image, kernel3, None, iterations=1)

    res1 = np.where((depth_image <= img1), 255, 0)
    res2 = np.where((img1 <= img2), 255, 0)
    res3 = np.where((img2 <= img3), 255, 0)

    res1 = np.where((res1 == res2), res1, 0)
    res2 = np.where((res2 == res3), res2, 0)
    res1 = np.where((res1 == res2), res1, 0)
    res1 = res1.astype(np.uint8)

    depth_image = np.where((res1 == 255), depth_image, 0)
    depth_image = np.where((depth_image == 5000), 0, depth_image)
    z = depth_image.flatten() / 1000



    x = np.multiply(x, z)
    y = np.multiply(y, z)

    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]

    return x, y, z

def diamond(n):
    res = np.zeros((2*n + 1, 2 * n + 1), np.uint8)
    for i in range(2*n + 1):
        if i <= n:
            y1 = n - i
            y2 = n + i
        else:
            y1 = i - n
            y2 = 3*n - i
        res[y1, i] = 1
        res[y2, i] = 1

    return res

def triangle_mask(n):
    res = np.zeros((2*n + 1, 2*n + 1), np.uint8)
    for i in range(2*n + 1):
        margin = int((i + 1) / 2)
        for j in range(margin, 2*n+1 - margin):
            res[i, j] = 1
    return res

def convert_depth_frame_to_pointcloud4(depth_image, camera_intrinsics):
    """
	Convert the depthmap to a 3D point cloud

	Parameters:
	-----------
	depth_frame 	 	 : rs.frame()
						   The depth_frame containing the depth map
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed

	Return:
	----------
	x : array
		The x values of the pointcloud in meters
	y : array
		The y values of the pointcloud in meters
	z : array
		The z values of the pointcloud in meters

	"""

    [height, width] = depth_image.shape

    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx) / camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy) / camera_intrinsics.fy

    kernel1 = diamond(9)
    kernel2 = diamond(19)
    kernel3 = diamond(29)

    depth_image = np.where((depth_image == 0), 5000, depth_image)
    img1 = cv2.erode(depth_image, kernel1, None, iterations=1)
    img2 = cv2.erode(depth_image, kernel2, None, iterations=1)
    img3 = cv2.erode(depth_image, kernel3, None, iterations=1)

    res1 = np.where((depth_image <= img1), 255, 0)
    res2 = np.where((img1 <= img2), 255, 0)
    res3 = np.where((img2 <= img3), 255, 0)

    res1 = np.where((res1 == res2), res1, 0)
    res2 = np.where((res2 == res3), res2, 0)
    res1 = np.where((res1 == res2), res1, 0)
    res1 = res1.astype(np.uint8)

    depth_image = np.where((res1 == 255), depth_image, 0)
    depth_image = np.where((depth_image == 5000), 0, depth_image)
    z = depth_image.flatten() / 1000



    x = np.multiply(x, z)
    y = np.multiply(y, z)

    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]

    return x, y, z

def convert_depth_frame_to_pointcloud5(depth_image, camera_intrinsics):
    [height, width] = depth_image.shape

    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx) / camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy) / camera_intrinsics.fy

    depth_image = np.where((depth_image == 0), 5000, depth_image)
    smooth_img = cv2.medianBlur(depth_image, 5)

    #kernel = triangle_mask(17)
    kernel = np.ones((51, 45), np.uint8)
    #dil_img = cv2.dilate(depth_image, kernel, None, iterations=1)
    erd_img = cv2.erode(smooth_img, kernel, None, iterations=1)

    #dil_img = dil_img - depth_image
    erd_img = smooth_img - erd_img

    #_, dil_img = cv2.threshold(dil_img, 0, 255, cv2.THRESH_BINARY_INV)
    #erd_img = np.where((dil_img > 0), 255, erd_img)
    _, erd_img = cv2.threshold(erd_img, 0, 255, cv2.THRESH_BINARY_INV)

    smooth_img2 = np.where((erd_img == 255), smooth_img, 0)
    smooth_img2 = np.where((depth_image == 5000), 0, smooth_img2)
    z2 = smooth_img2.flatten() / 1000

    smooth_img1 = np.where((depth_image == 5000), 0, smooth_img)
    z1 = smooth_img1.flatten() / 1000

    x2 = np.multiply(x, z2)
    y2 = np.multiply(y, z2)
    x2 = x2[np.nonzero(z2)]
    y2 = y2[np.nonzero(z2)]
    z2 = z2[np.nonzero(z2)]

    x1 = np.multiply(x, z1)
    y1 = np.multiply(y, z1)
    x1 = x1[np.nonzero(z1)]
    y1 = y1[np.nonzero(z1)]
    z1 = z1[np.nonzero(z1)]

    return [x2, y2, z2], [x1, y1, z1]

def get_corner_3d(depth_image, camera_intrinsics):
    return [0, 0, 0]

def convert_pointcloud_to_depth(pointcloud, camera_intrinsics):
    """
	Convert the world coordinate to a 2D image coordinate

	Parameters:
	-----------
	pointcloud 	 	 : numpy array with shape 3xN

	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed

	Return:
	----------
	x : array
		The x coordinate in image
	y : array
		The y coordiante in image

	"""

    assert (pointcloud.shape[0] == 3)
    x_ = pointcloud[0, :]
    y_ = pointcloud[1, :]
    z_ = pointcloud[2, :]

    m = x_[np.nonzero(z_)] / z_[np.nonzero(z_)]
    n = y_[np.nonzero(z_)] / z_[np.nonzero(z_)]

    x = m * camera_intrinsics.fx + camera_intrinsics.ppx
    y = n * camera_intrinsics.fy + camera_intrinsics.ppy

    return x, y


def get_boundary_corners_2D(points):
    """
	Get the minimum and maximum point from the array of points

	Parameters:
	-----------
	points 	 	 : array
						   The array of points out of which the min and max X and Y points are needed

	Return:
	----------
	boundary : array
		The values arranged as [minX, maxX, minY, maxY]

	"""
    padding = 0.05
    if points.shape[0] == 3:
        assert (len(points.shape) == 2)
        minPt_3d_x = np.amin(points[0, :])
        maxPt_3d_x = np.amax(points[0, :])
        minPt_3d_y = np.amin(points[1, :])
        maxPt_3d_y = np.amax(points[1, :])

        boudary = [minPt_3d_x - padding, maxPt_3d_x + padding, minPt_3d_y - padding, maxPt_3d_y + padding]

    else:
        raise Exception("wrong dimension of points!")

    return boudary


def get_clipped_pointcloud(pointcloud, boundary):
    """
	Get the clipped pointcloud withing the X and Y bounds specified in the boundary

	Parameters:
	-----------
	pointcloud 	 	 : array
						   The input pointcloud which needs to be clipped
	boundary      : array
										The X and Y bounds

	Return:
	----------
	pointcloud : array
		The clipped pointcloud

	"""
    assert (pointcloud.shape[0] >= 2)
    pointcloud = pointcloud[:, np.logical_and(pointcloud[0, :] < boundary[1], pointcloud[0, :] > boundary[0])]
    pointcloud = pointcloud[:, np.logical_and(pointcloud[1, :] < boundary[3], pointcloud[1, :] > boundary[2])]
    return pointcloud


def get_clipped_pointcloud2(pointcloud, sx=-0.4, ex=0.8, sy=-0.12, ey=0.42):
    assert (pointcloud.shape[0] >= 2)
    pointcloud = pointcloud[:, np.logical_and(pointcloud[0, :] < ex, pointcloud[0, :] > sx)]
    pointcloud = pointcloud[:, np.logical_and(pointcloud[1, :] < ey, pointcloud[1, :] > sy)]
    return pointcloud



