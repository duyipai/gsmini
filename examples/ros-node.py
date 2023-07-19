import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from sensor_msgs.msg import Image

from gsmini import gs3drecon, gsdevice


def main():
    rospy.init_node("gsminiros", anonymous=True)

    # Set flags
    DEVICE = "cuda"
    MASK_MARKERS_FLAG = True
    CALCULATE_DEPTH_FLAG = True
    CALCULATE_SHEAR_FLAG = True
    SHOW_NOW = False

    # the device ID can change after unplugging and changing the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    cam_id = gsdevice.get_camera_id("GelSight Mini")
    dev = gsdevice.Camera(
        cam_id,
        calcDepth=CALCULATE_DEPTH_FLAG,
        calcShear=CALCULATE_SHEAR_FLAG,
        device=DEVICE,
        maskMarkersFlag=MASK_MARKERS_FLAG,
    )

    cvbridge = CvBridge()
    image_pub = rospy.Publisher("/gsmini_image", Image, queue_size=1)

    if CALCULATE_DEPTH_FLAG:
        depth_pub = rospy.Publisher("/gsmini_depth", numpy_msg(Floats), queue_size=1)

        """ use this to plot just the 3d """
        if SHOW_NOW:
            vis3d = gs3drecon.Visualize3D(dev.imgw, dev.imgh, "", dev.mmpp)

    if CALCULATE_SHEAR_FLAG:
        shear_pub = rospy.Publisher("/gsmini_shear", numpy_msg(Floats), queue_size=1)

    rate = rospy.Rate(14)
    while not rospy.is_shutdown():
        rate.sleep()

        f1 = dev.get_image()
        dev.process_image(f1)

        """ publish images """
        image_pub.publish(cvbridge.cv2_to_imgmsg(f1, encoding="passthrough"))

        if CALCULATE_DEPTH_FLAG:
            dm = dev.get_depth()
            depth_pub.publish(dm.flatten())

        if CALCULATE_SHEAR_FLAG:
            markers = dev.get_markers()
            shear = np.stack((dev.get_initial_markers(), markers), axis=1)
            shear_pub.publish(shear.flatten())
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
