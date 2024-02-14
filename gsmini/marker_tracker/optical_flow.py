import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.signal import fftconvolve


def gkern(l=5, sig=1.0):
    """ creates gaussian kernel with side length l and a sigma of sig """
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)


def normxcorr2(template, image, mode="same"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array    """
    # If this happens, it is probably a mistake
    if (
        np.ndim(template) > np.ndim(image)
        or len(
            [i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]
        )
        > 0
    ):
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")
    template = template - np.mean(template)
    image = image - np.mean(image)
    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)
    image = fftconvolve(np.square(image), a1, mode=mode) - np.square(
        fftconvolve(image, a1, mode=mode)
    ) / (np.prod(template.shape))
    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0
    template = np.sum(np.square(template))

    out = out / np.sqrt(image * template)
    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    return out


def find_marker(gray):
    adjusted = cv2.convertScaleAbs(gray, alpha=3, beta=0)
    mask = cv2.inRange(adjusted, 255, 255)
    """ normalized cross correlation """
    # template = gkern(l=20, sig=5)
    template = gkern(l=20, sig=5)

    nrmcrimg = normxcorr2(template, mask)
    """""" """""" """""" """""" """""" """"""
    a = nrmcrimg
    b = 2 * ((a - a.min()) / (a.max() - a.min())) - 1
    b = (b - b.min()) / (b.max() - b.min())
    mask = np.asarray(b < 0.5)  # 0.5
    return (mask * 255).astype("uint8")


def find2dpeaks(res):
    """
    Create masks for all the dots. find 2D peaks.
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    img3 = res
    neighborhood_size = 20
    threshold = 0
    data_max = maximum_filter(img3, neighborhood_size)
    maxima = img3 == data_max
    data_min = minimum_filter(img3, neighborhood_size)
    diff = (data_max - data_min) > threshold
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    xy = np.array(ndimage.center_of_mass(img3, labeled, range(1, num_objects + 1)))
    return xy


class MarkerTracking:
    def __init__(self, init_frame, N=7, M=9, params=None):
        self.N = N
        self.M = M
        self.NUM_MKS = N * M
        # Parameters for Lucas Kanade optical flow
        self.init_frame = init_frame
        self.lk_params = dict(
            winSize=(100, 100),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 2),
            flags=0,
        )

        init_framegray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
        init_framegray = cv2.adaptiveThreshold(
            init_framegray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,
            10,
        )

        # find mask
        mask = find_marker(init_framegray)
        cv2.imwrite("mask.jpg", mask)
        # find marker centers
        self.mc0 = find2dpeaks(mask)
        self.mc0[:, [0, 1]] = self.mc0[:, [1, 0]]  # Mx2

        self.MK_SCALE = 10
        self.original = np.zeros((self.NUM_MKS, 2, 2))
        # p2 = M x [x,y]
        # self.mc0 = M x [x,y]
        # self.mc0
        # xy1 = MK_SCALE * (p2 - self.mc0) + self.mc0 (M x [x,y])

    def get_flow(self, frame, reset=False, debug_img=False):
        p2, st2, err2 = cv2.calcOpticalFlowPyrLK(
            self.init_frame, frame, self.mc0.astype("float32"), None, **self.lk_params
        )
        curr = np.zeros((self.NUM_MKS, 2, 2))
        curr[:, 0, :] = self.mc0
        curr[:, 1, :] = self.MK_SCALE * (p2 - self.mc0) + self.mc0
        curr[:, 1, :] = curr[:, 1, :] - self.original[:, 1, :] + self.original[:, 0, :]
        if reset:
            self.original[:, 0, :] = self.mc0
            self.original[:, 1, :] = self.MK_SCALE * (p2 - self.mc0) + self.mc0

        if debug_img:
            for mk in range(curr.shape[0]):
                curr_i = curr.copy().astype(np.int32)
                cv2.arrowedLine(
                    frame,
                    (curr_i[mk, 0, 0], curr_i[mk, 0, 1]),
                    (curr_i[mk, 1, 0], curr_i[mk, 1, 1]),
                    (0, 255, 255),
                    thickness=1,
                    tipLength=0.2,
                )
            cv2.imwrite("debug_img.jpg", frame)
        return curr

    @property
    def marker_shape(self):
        return (self.N, self.M)
