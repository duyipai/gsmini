import copy
import math
from enum import Enum

import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage import measure
from skimage.morphology import closing, disk

from .fit_grid import fit_grid
from .normxcorr2 import normxcorr2


class GridStyle(Enum):
    NOBORDER = 0
    ALL = 1


class MarkerTracker:
    def __init__(self, marker_image, params=None):
        if params is None:
            self.params = {"GridStyle": GridStyle.NOBORDER, "DoPlot": False}
        else:
            self.params = params

        img = marker_image

        filtsig = 20
        bwthresh = -0.05

        mxim = np.max(img.astype(float), axis=2)
        ydim, xdim = mxim.shape

        # High-pass filter
        # fsg = filtsig
        # fsz = 2 * round(3 * fsg) + 1
        # gf = gaussian(fsz, fsg)

        gf = gauss_signal(filtsig)
        filtim = gauss_filt(mxim, gf)
        # filtim = cv2.filter2D(mxim, -1, gf, borderType=cv2.BORDER_REPLICATE)

        hpim = mxim - filtim

        se = disk(3)
        bw = hpim < bwthresh
        # lb = regionprops_label(closing(bw, selem=se))
        label_image = measure.label(closing(bw, footprint=se))

        # Set minimum size threshold
        min_size = 20
        max_size = 500

        # Remove connected components smaller than the threshold
        clean_labels = np.zeros_like(label_image)
        for i in range(1, label_image.max() + 1):
            blobsize = (label_image == i).sum()
            if blobsize >= min_size and blobsize <= max_size:
                clean_labels[label_image == i] = i

        marker_mask = np.where(clean_labels != 0, 1, 0)
        # cv2.imshow('original_mask', marker_mask.astyp(float))

        all_props = measure.regionprops(clean_labels, mxim)
        areas = np.array([prop.area for prop in all_props])
        centers = np.array([prop.centroid for prop in all_props])
        intensities = np.array([prop.intensity_mean for prop in all_props])

        # Sort the centers and get corresponding indices
        # sorted_indices = np.lexsort((centers[:, 1], centers[:, 0]))
        # sorted_centers = centers[sorted_indices]
        sorted_indices = self.sort_centers(centers)
        sorted_centers = centers[sorted_indices]
        sorted_areas = areas[sorted_indices]
        sorted_intensities = intensities[sorted_indices]

        # Find the grid spacing
        gsp = self.estimate_grid_spacing(sorted_centers)

        new_centers = sorted_centers
        new_areas = sorted_areas
        new_intensities = sorted_intensities

        # NOBORDER will restrict to uniform grid
        if self.params["GridStyle"] == GridStyle.ALL:
            (
                num_rows,
                num_cols,
                row_coordinates,
                col_coordinates,
            ) = self.assign_coordinates(sorted_centers)
        else:
            # Point needs to be minsp away from other points and minpd away from image boundary
            minpd = gsp / 3
            minsp = gsp / 2
            good_centers = []
            good_indices = np.empty((0,), dtype=int)
            for c in range(sorted_centers.shape[0]):
                pt = sorted_centers[c, :]
                if (pt[1] > minpd and pt[1] < xdim - minpd) and (
                    pt[0] > minpd and pt[0] < xdim - minpd
                ):
                    if len(good_centers) > 0:
                        lastpts = np.asarray(good_centers)
                        dsts = np.sqrt(
                            np.square(pt[1] - lastpts[:, 1])
                            + np.square(pt[0] - lastpts[:, 0])
                        )
                        if not np.any(dsts < minsp):
                            good_centers.append(pt)
                            good_indices = np.append(good_indices, c)
                    else:
                        good_centers.append(pt)
                        good_indices = np.append(good_indices, c)

            new_centers = np.asarray(good_centers)
            new_areas = sorted_areas[good_indices]
            new_intensities = sorted_intensities[good_indices]

            # fit centers to a grid
            gridpts, gridw = fit_grid(new_centers, gsp)

            # get the grid coordinates
            gridct = gridw / gsp
            gridct[:, 1] = gridct[:, 1] - np.min(gridct[:, 1])
            gridct[:, 0] = gridct[:, 0] - np.min(gridct[:, 0])
            num_cols = int(max(gridct[:, 0]) + 1)
            num_rows = int(max(gridct[:, 1]) + 1)
            row_coordinates = (np.round(gridct[:, 1])).astype("int")
            col_coordinates = (np.round(gridct[:, 0])).astype("int")

        print(f"Number of rows, cols: {num_rows}, {num_cols}")
        # print("Row coordinates:", row_coordinates)
        # print("Column coordinates:", col_coordinates)

        nct = new_centers.shape[0]
        # Estimate dot radius
        radii = np.zeros(nct)
        for i in range(nct):
            p = new_centers[i, :]
            radii[i] = np.sqrt(new_areas[i] / np.pi)

        # Save marker data in struct
        self.xdim = mxim.shape[1]
        self.ydim = mxim.shape[0]
        self.gridsz = [num_cols, num_rows]
        self.gridsp = gsp
        self.marker_mask = marker_mask
        self.initial_marker_coord = [col_coordinates, row_coordinates]
        self.initial_marker_center = new_centers
        self.marker_radius = radii
        self.marker_blackpt = new_intensities
        self.marker_center = new_centers
        self.marker_lastpos = new_centers
        self.marker_currentpos = new_centers

    def create_markermask(self, img, centers, radius):
        markermask = np.zeros((img.shape[0], img.shape[1]))

        for c in range(len(centers)):
            cv2.circle(
                markermask,
                (int(centers[c, 1]), int(centers[c, 0])),
                int(radius[c]),
                color=(255, 255, 255),
                thickness=-1,
            )
        cv2.imshow("mask", markermask)

        return markermask

    def sort_centers(self, dot_centers):
        sorted_indices = np.lexsort((dot_centers[:, 1], dot_centers[:, 0]))
        sorted_dot_centers = dot_centers[sorted_indices]

        # Extract the x and y coordinates of the sorted dot centers
        x_coords = [dot[0] for dot in sorted_dot_centers]
        y_coords = [dot[1] for dot in sorted_dot_centers]

        # Calculate the number of rows and columns
        lenx = len(x_coords)

        # Loop through each dot center and assign grid coordinates
        lx = 0
        new_sorted_indices = np.empty((0,), dtype=int)

        while lx < lenx - 1:
            xvals = []
            yvals = []
            old_sorted_indices = np.empty((0,), dtype=int)
            x0 = x_coords[lx]
            y0 = y_coords[lx]
            xvals.append(x0)
            yvals.append(y0)
            old_sorted_indices = np.append(old_sorted_indices, sorted_indices[lx])
            lx = lx + 1
            x1 = x_coords[lx]
            y1 = y_coords[lx]
            while (x1 - x0) < 10 and lx < lenx - 1:
                old_sorted_indices = np.append(old_sorted_indices, sorted_indices[lx])
                xvals.append(x1)
                yvals.append(y1)
                lx = lx + 1
                x0 = x1
                x1 = x_coords[lx]
                y1 = y_coords[lx]
            if lx == lenx - 1:
                old_sorted_indices = np.append(old_sorted_indices, sorted_indices[lx])
                xvals.append(x_coords[lx])
                yvals.append(y_coords[lx])
            if len(yvals) > 5:
                sorted_yindx = np.argsort(yvals)
                old_sorted = old_sorted_indices[sorted_yindx]
                new_sorted_indices = np.append(new_sorted_indices, old_sorted)

        return new_sorted_indices

    def assign_coordinates(self, dot_centers):
        # Sort the dot centers by their x and y coordinates
        # sorted_dot_centers = sorted(dot_centers, key=lambda x: (x[0], x[1]))
        x_coords = [dot[0] for dot in dot_centers]
        y_coords = [dot[1] for dot in dot_centers]

        # Calculate the number of rows and columns
        ncoords = len(y_coords)

        # Create dictionaries to store the mapping from dot center to grid coordinates
        row_coordinates = np.zeros(ncoords)
        col_coordinates = np.zeros(ncoords)

        idx = 0
        idy = 0
        n = 0
        num_rows = 0
        num_cols = 0
        x0 = x_coords[idx]
        col_coordinates[n] = idx
        row_coordinates[n] = idy
        while n < ncoords - 1:
            col_coordinates[n] = idx
            row_coordinates[n] = idy
            x1 = x_coords[n + 1]
            while (x1 - x0) < 15 and n < ncoords - 2:
                n = n + 1
                idy = idy + 1
                col_coordinates[n] = idx
                row_coordinates[n] = idy
                x0 = x1
                x1 = x_coords[n + 1]
                # if lx < lenx - 2:
                #     x1 = x_coords[lx+1]
            n = n + 1
            if idx == 0:
                num_rows = n
            yind = idy + 1
            idy = 0
            idx = idx + 1
            num_cols = idx
            x0 = x1

        col_coordinates[n] = int(idx - 1)
        row_coordinates[n] = int(yind)

        return num_cols, num_rows, col_coordinates, row_coordinates

    def estimate_grid_spacing(self, centers):
        N = centers.shape[0]
        dsts = np.zeros(4 * N)
        for i in range(N):
            p = centers[i, :]
            d = np.sqrt((centers[:, 1] - p[1]) ** 2 + (centers[:, 0] - p[0]) ** 2)
            srtd = np.sort(d)
            dsts[4 * i] = srtd[1]
            dsts[4 * i + 1] = srtd[2]
            dsts[4 * i + 2] = srtd[3]
            dsts[4 * i + 3] = srtd[4]
        gsp = np.median(dsts)
        return gsp


def gauss_signal(sigma):
    # Define the size and sigma of the Gaussian filter
    fsg = sigma
    fsz = 2 * round(3 * fsg) + 1

    # Create a 1D Gaussian kernel using numpy
    k = np.exp(-np.arange(-(fsz - 1) / 2, (fsz - 1) / 2 + 1) ** 2 / (2 * fsg**2))

    # Normalize the kernel
    k = k / np.sum(k)

    # Convert the 1D kernel to a 2D filter by taking the outer product
    gf = np.outer(k, k)

    # Normalize the filter
    gf = gf / np.sum(gf)

    return gf


def gauss_filt(input_img, gf_img):
    # Convert the input image to grayscale
    # gray_img = np.mean(input_img, axis=2)

    # Apply Gaussian smoothing with the gaussian_image as filter
    smoothed_image = convolve(input_img, gf_img)

    return smoothed_image


if __name__ == "__main__":
    cp = cv2.VideoCapture("data/mini_example.avi")
    ret, img = cp.read()
    for i in range(10):
        ret, img = cp.read()

    # cv2.imwrite('f0.png', img)
    # cv2.imshow('img', img)
    # cv2.waitKey()

    params = {"GridStyle": GridStyle.NOBORDER, "DoPlot": True}

    img = np.float32(img) / 255.0
    mtrack = MarkerTracker(img, params)

    print(mtrack)
