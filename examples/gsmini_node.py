#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from gsmini import gs3drecon, gsdevice
from gsmini.msg import Shear


def main():
    rospy.init_node("gsminiros")

    # Set flags
    DEVICE = rospy.get_param("~device", "cuda")  # robot radius in grid units
    CAMERA_DEVICE = rospy.get_param("~camera_device")
    MASK_MARKERS_FLAG = rospy.get_param("~mask_markers", True)
    CALCULATE_SHEAR_FLAG = rospy.get_param("~calculate_shear", True)
    CALCULATE_DEPTH_FLAG = rospy.get_param("~calculate_depth", False)
    SHOW_NOW = rospy.get_param("~show_now", False)

    dev = gsdevice.Camera(
        CAMERA_DEVICE,
        calcDepth=CALCULATE_DEPTH_FLAG,
        calcShear=CALCULATE_SHEAR_FLAG,
        device=DEVICE,
        maskMarkersFlag=MASK_MARKERS_FLAG,
    )

    cvbridge = CvBridge()
    image_pub = rospy.Publisher("gsmini_image", Image, queue_size=1)

    if CALCULATE_DEPTH_FLAG:
        depth_pub = rospy.Publisher("gsmini_depth", Image, queue_size=1)

        """ use this to plot just the 3d """
        if SHOW_NOW:
            vis3d = gs3drecon.Visualize3D(dev.imgw, dev.imgh, "", dev.mmpp)

    if CALCULATE_SHEAR_FLAG:
        shear_pub = rospy.Publisher("gsmini_shear", Shear, queue_size=1)

    rate = rospy.Rate(14)
    while not rospy.is_shutdown():
        rate.sleep()

        f1 = dev.get_image()
        dev.process_image(f1)

        """ publish images """
        image_pub.publish(cvbridge.cv2_to_imgmsg(f1, encoding="bgr8"))

        if CALCULATE_DEPTH_FLAG:
            dm = dev.get_depth()
            depth_pub.publish(cvbridge.cv2_to_imgmsg(dm, encoding="32FC1"))

        if CALCULATE_SHEAR_FLAG:
            shear_np = dev.get_shear()
            shear_msg = Shear()
            shear_msg.header.stamp = rospy.Time.now()
            shear_msg.n = dev.marker_shape[0]
            shear_msg.m = dev.marker_shape[1]
            shear_msg.header.frame_id = rospy.get_name()
            shear_msg.initial = shear_np[:, 0, :].flatten()
            shear_msg.markers = shear_np[:, 1, :].flatten()
            shear_pub.publish(shear_msg)
        """ Display the results """
        if SHOW_NOW:
            cv2.imshow("Image", f1)
            if CALCULATE_DEPTH_FLAG:
                vis3d.update(dm)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    dev.disconnect()


if __name__ == "__main__":
    main()
