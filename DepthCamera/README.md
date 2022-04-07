# Depth Camera + Object Detector to capture Heigh and Distance
This is a simple script example used to capture the depth information from a RealSense camera and associate it to the pixel values of the Detected Objects by Yolov5 COCO.
Therefore, it allwos to obtain an approximate (x, y, z) coordinates in real-world frame to capture the real distance in meters. It also can compute the height of the bounding boxes in the real world, thus obtaining an approximation of the height of people or objects, for instance.

## Install and Preparation of Environment
This is a python code, not a Jupyter script. Therefore, in order to run it, it needs first to:

Install Python 3.7 (higher version than that causes some errors when installing the pyrealsense2)

Install the drivers for the RealSense: follow the steps indicated in their [official documentation](https://github.com/IntelRealSense/librealsense)
Install the RealSense library to be managed with python. 

```bash
pip install pyrealsense2
```

Install the Yolov5 as Object Detector and prepare the environment.
```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

The  [Capture heigh and depth](https://github.com/Evm7/Tutorials-Computer-Vision/blob/master/DepthCamera/CaptureHeightandDepth.py) script shall be placed inside the Yolov5 directory.

## Run
Run the scripts without any arguments.
