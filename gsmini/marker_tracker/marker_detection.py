import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy import ndimage
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
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """
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


def init(frame):
    RESCALE = 1.0
    return cv2.resize(frame, (0, 0), fx=1.0 / RESCALE, fy=1.0 / RESCALE)


def preprocessimg(img):
    """
    Pre-processing image to remove noise
    """
    dotspacepx = 36
    ### speckle noise denoising
    # dst = cv2.fastNlMeansDenoising(img_gray, None, 9, 15, 30)
    ### adaptive histogram equalizer
    # clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(10, 10))
    # equalized = clahe.apply(img_gray)
    ### Gaussian blur
    # gsz = 2 * round(3 * mindiameterpx / 2) + 1
    gsz = 2 * round(0.75 * dotspacepx / 2) + 1
    blur = cv2.GaussianBlur(img, (51, 51), gsz / 6)
    #### my linear varying filter
    x = np.linspace(3, 1.5, img.shape[1])
    y = np.linspace(3, 1.5, img.shape[0])
    xx, yy = np.meshgrid(x, y)
    mult = blur * yy
    ### adjust contrast
    res = cv2.convertScaleAbs(blur, alpha=2, beta=0)
    return res


def find_marker(frame):
    ##### masking techinique for dots on mini
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # mask = cv2.inRange(gray, 5, 55)

    gray = frame[:, :, 1]  ### use only the green channel
    im_blur_3 = cv2.GaussianBlur(gray, (3, 3), 5)
    im_blur_8 = cv2.GaussianBlur(gray, (15, 15), 5)
    im_blur_sub = im_blur_8 - im_blur_3 + 128
    mask = cv2.inRange(im_blur_sub, 140, 255)

    # ''' normalized cross correlation '''
    template = gkern(l=20, sig=3)
    nrmcrimg = normxcorr2(template, mask)
    # ''''''''''''''''''''''''''''''''''''
    a = nrmcrimg
    mask = np.asarray(a > 0.1)
    mask = (mask).astype("uint8")

    return mask


def marker_center(mask, frame):
    img3 = mask
    neighborhood_size = 10
    # threshold = 40 # for r1.5
    threshold = 0  # for mini
    data_max = maximum_filter(img3, neighborhood_size)
    maxima = img3 == data_max
    data_min = minimum_filter(img3, neighborhood_size)
    diff = (data_max - data_min) > threshold
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    MarkerCenter = np.array(
        ndimage.center_of_mass(img3, labeled, range(1, num_objects + 1))
    )
    MarkerCenter[:, [0, 1]] = MarkerCenter[:, [1, 0]]
    for i in range(MarkerCenter.shape[0]):
        x0, y0 = int(MarkerCenter[i][0]), int(MarkerCenter[i][1])
        cv2.circle(mask, (x0, y0), color=(0, 0, 0), radius=1, thickness=1)
    return MarkerCenter


def draw_flow(frame, flow):
    Ox, Oy, Cx, Cy, Occupied = flow

    dx = np.mean(np.abs(np.asarray(Ox) - np.asarray(Cx)))
    dy = np.mean(np.abs(np.asarray(Oy) - np.asarray(Cy)))
    dnet = np.sqrt(dx ** 2 + dy ** 2)
    print(dnet * 0.075, "\n")

    K = 1
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            pt1 = (int(Ox[i][j]), int(Oy[i][j]))
            pt2 = (
                int(Cx[i][j] + K * (Cx[i][j] - Ox[i][j])),
                int(Cy[i][j] + K * (Cy[i][j] - Oy[i][j])),
            )
            color = (0, 0, 255)
            if Occupied[i][j] <= -1:
                color = (127, 127, 255)
            cv2.arrowedLine(frame, pt1, pt2, color, 2, tipLength=0.25)


def warp_perspective(img):

    TOPLEFT = (175, 230)
    TOPRIGHT = (380, 225)
    BOTTOMLEFT = (10, 410)
    BOTTOMRIGHT = (530, 400)

    WARP_W = 215
    WARP_H = 215

    points1 = np.float32([TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT])
    points2 = np.float32([[0, 0], [WARP_W, 0], [0, WARP_H], [WARP_W, WARP_H]])

    matrix = cv2.getPerspectiveTransform(points1, points2)

    result = cv2.warpPerspective(img, matrix, (WARP_W, WARP_H))

    return result


def init_HSR(img):
    DIM = (640, 480)
    img = cv2.resize(img, DIM)

    K = np.array(
        [
            [225.57469247811056, 0.0, 280.0069549918857],
            [0.0, 221.40607131318117, 294.82435570493794],
            [0.0, 0.0, 1.0],
        ]
    )
    D = np.array(
        [
            [0.7302503082668154],
            [-0.18910060205317372],
            [-0.23997727800712282],
            [0.13938490908400802],
        ]
    )
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, DIM, cv2.CV_16SC2
    )
    undistorted_img = cv2.remap(
        img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    return warp_perspective(undistorted_img)
