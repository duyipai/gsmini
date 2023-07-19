import time

import cv2
import numpy as np

from gsmini import gs3drecon, gsdevice


def main():
    # Set flags
    SAVE_VIDEO_FLAG = False
    DEVICE = "cuda"
    MASK_MARKERS_FLAG = True
    CALCULATE_DEPTH_FLAG = True
    CALCULATE_SHEAR_FLAG = True

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

    if SAVE_VIDEO_FLAG:
        # Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        file_path = "./3dnnlive.mov"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(file_path, fourcc, 60, (160, 120), isColor=True)

    print("press q on image to exit")

    if CALCULATE_DEPTH_FLAG:
        # """ use this to plot just the 3d """
        vis3d = gs3drecon.Visualize3D(dev.imgw, dev.imgh, "", dev.mmpp)

    if CALCULATE_SHEAR_FLAG:
        color = np.random.randint(0, 255, (100, 3))
        init_markers = dev.get_initial_markers()
    start_time = time.time()
    count = 0
    avg = 50
    while True:
        f1 = dev.get_image()
        dev.process_image(f1)
        if CALCULATE_SHEAR_FLAG:
            markers = dev.get_markers()
            for i, new in enumerate(markers):
                a, b = new.ravel()
                ix = int(init_markers[i, 0])
                iy = int(init_markers[i, 1])
                f1 = cv2.arrowedLine(
                    f1,
                    (ix, iy),
                    (int(a), int(b)),
                    (255, 255, 255),
                    thickness=1,
                    line_type=cv2.LINE_8,
                    tipLength=0.15,
                )
                f1 = cv2.circle(f1, (int(a), int(b)), 5, color[i].tolist(), -1)
        if CALCULATE_DEPTH_FLAG:
            dm = dev.get_depth()
            vis3d.update(dm)

        cv2.imshow("Image", f1)

        count += 1
        if count > avg:
            print("fps", avg / (time.time() - start_time))
            start_time = time.time()
            count = 0
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if SAVE_VIDEO_FLAG:
            out.write(f1)
    dev.disconnect()


if __name__ == "__main__":
    main()
