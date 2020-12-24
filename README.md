# Corner detection in 2 mm accuracy of the box using multiple intel realsense cameras
There is a box detection example code that Intel Realsense provides.
But the error is larger than 1cm in general environment.
This script make it possible to detect the corners of box with high accuracy using four intel realsense cameras.
![calibration](https://github.com/araj89/MultiIntelRealsense_BoxCornerDetection/blob/master/Coordinates2.png)

## dependencies
 - OpenCV 4.2
 - pyrealsense2
 
## setup of cameras
The intel realsense cameras should be installed that can see 3 surface of the box.
  