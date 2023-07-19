# SDK for GelSight Mini robotic sensors

This repository is adapted from the official GelSight code (https://github.com/gelsightinc/gsrobotics), but refactored to maintain clean, high-performance and user-friendly. Note that this repository is specifically for the Mini sensor. The idea is to only expose necessary functionalities to the users while make all the other processings under the hood. Examples to obtain the sensor data for visualization and ros are provided in the examples folder.

## Prerequisites

    Python 3.8 or above

## Install python libraries
    pip3 install .
    or 
    pip3 install . --upgrade


## Device setup

The camera on Mini is a USB camera. You can change the camera parameters using any app or library that can control UVC cameras. On Ubuntu, one such popular library is [v4l2-ctl](https://manpages.ubuntu.com/manpages/bionic/man1/v4l2-ctl.1.html).

## Linux setup
To install this library on ubuntu run, 

    sudo apt-get update
    sudo apt-get -y install v4l-utils

Refer to file config/mini_set_cam_params.sh present in this repository to view/edit all the available camera parameters. 
You can list the devices by running:

    v4l2-ctl --list-devices


To set the camera parameters listed in mini_set_cam_params.sh file, run, 

    sudo ./config/mini_set_cam_params.sh 2

Note the scripts takes the camera device id (0, or 1, or 2, or 3,.. etc), as the first argument. In most cases when you have one Mini connected to 
you computers, the device ID is usually 2, because the webcam on your computer is always on device ID 0.

## Windows setup

The script tries to find the correct camera device id on Windows.
You may need to change the following line in show3d.py and showimages.py

    dev = gsdevice.Camera(1)


## View Signal (Raw image, Shear, and Normal)
    cd examples
    python3 show.py

## ROS node
    cd examples
    python ros_node.py

To obtain the expected results from the algorithms implemented on the raw image, please set the default camera parameters present in mini_set_cam_params.sh.
