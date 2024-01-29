import os
import re
from threading import Thread

import cv2
import numpy as np

from . import gs3drecon
from .markertracker import MarkerTracker
import subprocess


def get_camera_id(camera_name):
    # Execute the command and capture its output
    output = subprocess.check_output("v4l2-ctl --list-devices", shell=True, text=True)

    # Split the output into lines
    lines = output.split("\n")

    # Initialize variables
    camera_found = False
    video_devices = []

    # Iterate over the lines
    for line in lines:
        # Check if the current line contains the camera name
        if camera_name in line:
            camera_found = True
            continue

        # If camera is found, look for video devices
        if camera_found:
            match = re.search(r"/dev/video(\d+)", line)
            if match:
                video_devices.append(int(match.group(1)))
            else:
                # No more video devices for the camera, break the loop
                break

    # Filter for even video devices
    even_video_devices = [dev for dev in video_devices if dev % 2 == 0]

    assert (
        len(even_video_devices) == 1
    ), "Video device not found or multiple devices found. Only one video device should be connected."

    return even_video_devices[0]


def resize_crop_mini(img, imgw, imgh):
    border_size_x, border_size_y = (
        int(img.shape[0] * (1 / 7)),
        int(np.floor(img.shape[1] * (1 / 7))),
    )  # remove 1/7th of border from each size
    img = img[
        border_size_x : img.shape[0] - border_size_x,
        border_size_y : img.shape[1] - border_size_y,
        :,
    ]
    img = cv2.resize(
        img, (imgw, imgh), interpolation=cv2.INTER_AREA
    )  # final resize for 3d
    return img


class Camera:
    def __init__(
        self,
        dev_id,  # str means path to images, int means camera id
        calcDepth=False,
        calcShear=False,
        device="cpu",
        maskMarkersFlag=True,
        netPath=os.path.join(os.path.dirname(gs3drecon.__file__), "nnmini.pt"),
        mmpp=0.0625,
    ):
        # variable to store data
        self.imgw = 320
        self.imgh = 240
        self.mmpp = mmpp
        self._dm = None
        self._Ox = None
        self._Oy = None
        self._p0 = None
        self._dm_dirty = False
        self._shear_dirty = False
        self.dev_id = dev_id
        self.cam = None
        self.enableDepth = calcDepth
        self.maskMarkersFlag = maskMarkersFlag
        if self.enableDepth:
            self.nn = gs3drecon.Reconstruction3D(self.imgw, self.imgh)
            _ = self.nn.load_nn(netPath, device)
        self.enableShear = calcShear
        self.connect()

    def connect(self):
        if (
            type(self.dev_id) == str
        ):  # if dev_id is a string, then it is a path used to initialize the depth and marker
            import glob

            paths = glob.glob(os.path.join(self.dev_id, "*.jpg"))
            for path in paths:
                self._img = cv2.imread(path)
                if self.enableDepth:
                    self._dm = self.nn.get_depthmap(self._img, self.maskMarkersFlag)
        else:
            self.cam = cv2.VideoCapture(self.dev_id)
            if self.cam is None or not self.cam.isOpened():
                print("Warning: unable to open video source: ", self.dev_id)
            self._img = self.get_raw_image()
            if self.enableDepth:
                while self.nn.dm_zero_counter < 50:
                    ret, self._img = self.cam.read()
                    self._img = resize_crop_mini(self._img, self.imgw, self.imgh)
                    if ret:
                        self._dm = self.nn.get_depthmap(self._img, self.maskMarkersFlag)
        if self.enableShear:
            self._old_gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
            mtracker = MarkerTracker(np.float32(self._img) / 255.0)
            self._Ox = mtracker.initial_marker_center[:, 1]
            self._Oy = mtracker.initial_marker_center[:, 0]
            self._initial_markers = np.array(
                (self._Ox, self._Oy), np.float32
            ).T.reshape((-1, 2))
            self._nct = len(mtracker.initial_marker_center)
            self._lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
            self._p0 = np.array((self._Ox, self._Oy), np.float32).T.reshape((-1, 1, 2))
            # finished initializing p0
        if self.cam is not None:
            self._stop = False
            Thread(target=self._update_image).start()
        return

    def get_raw_image(self):
        for _ in range(10):  # flush out fist 10 frames to remove black frames
            ret, f0 = self.cam.read()
        ret, f0 = self.cam.read()
        if ret:
            f0 = resize_crop_mini(f0, self.imgw, self.imgh)
        else:
            print("ERROR! reading image from camera")

        return f0

    def get_image(self):
        return self._img.copy()

    def get_depth(self):
        if not self.enableDepth:
            print("ERROR! depth is not enabled")
            return None
        return self._dm.copy()

    def get_markers(self):
        if not self.enableShear:
            print("ERROR! shear is not enabled")
            return None
        return np.squeeze(self._p0.copy())

    def get_initial_markers(self):
        if not self.enableShear:
            print("ERROR! shear is not enabled")
            return None
        return self._initial_markers.copy()

    def process_image(self, img):
        if self.enableDepth:
            self._update_depth(img)
        if self.enableShear:
            self._update_shear(img)

    def _update_image(self):
        while not self._stop:
            ret, f0 = self.cam.read()
            if ret:
                f0 = resize_crop_mini(f0, self.imgw, self.imgh)
                self._img = f0

    def _update_depth(self, img):
        self._dm = self.nn.get_depthmap(img, self.maskMarkersFlag)

    def _update_shear(self, img):
        new_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            self._old_gray, new_gray, self._p0, None, **self._lk_params
        )
        if np.sum(st) < self._nct:
            print("all pts did not converge")
        else:
            self._p0 = p1.reshape(-1, 1, 2)
            self._old_gray = new_gray

    def disconnect(self):
        self._stop = True
