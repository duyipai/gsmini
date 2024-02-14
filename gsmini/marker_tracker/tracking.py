import numpy as np
import gsmini.cpptracker.marker_detection as marker_detection
import gsmini.cpptracker.find_marker as find_marker


def marker_calibration(frame, N, M):
    """
    Args:
    frame: the frame that contains the marker
    N, M: the row and column of the marker array
    Returns:
    x0_, y0_: the coordinate of upper-left marker
    dx_, dy_: the horizontal and vertical interval between adjacent markers
    """
    mask = marker_detection.find_marker(frame)
    mc = marker_detection.marker_center(mask, frame)
    mc_sorted1 = mc[mc[:, 0].argsort()]
    mc1 = mc_sorted1[:N]
    mc1 = mc1[mc1[:, 1].argsort()]

    mc_sorted2 = mc[mc[:, 1].argsort()]
    mc2 = mc_sorted2[:M]
    mc2 = mc2[mc2[:, 0].argsort()]

    x0_ = np.round(mc1[0][0])
    y0_ = np.round(mc1[0][1])
    dx_ = mc2[1, 0] - mc2[0, 0]
    dy_ = mc1[1, 1] - mc1[0, 1]

    return x0_, y0_, dx_, dy_


class MarkerTracking:
    def __init__(self, frame, N=7, M=9, FPS=25):
        self.N_ = N
        self.M_ = M
        self.FPS_ = FPS
        self.x0_, self.y0_, self.dx_, self.dy_ = marker_calibration(
            frame, self.N_, self.M_
        )
        self.matcher = find_marker.Matching(
            self.N_, self.M_, self.FPS_, self.x0_, self.y0_, self.dx_, self.dy_
        )

    def get_flow(self, frame):
        mask = marker_detection.find_marker(frame)
        mc = marker_detection.marker_center(mask, frame)
        self.matcher.init(mc)
        self.matcher.run()
        flow = self.matcher.get_flow()
        print(flow)
        """
        flow: (Ox, Oy, Cx, Cy, Occupied)
            Ox, Oy: N*M matrix, the x and y coordinate of each marker at frame 0
            Cx, Cy: N*M matrix, the x and y coordinate of each marker at current frame
            Occupied: N*M matrix, the index of the marker at each position, -1 means inferred.
                e.g. Occupied[i][j] = k, meaning the marker mc[k] lies in row i, column j.
        """

        Ox = np.array(flow[0]).reshape(self.N_, self.M_)
        Oy = np.array(flow[1]).reshape(self.N_, self.M_)

        Cx = np.array(flow[2]).reshape(self.N_, self.M_)
        Cy = np.array(flow[3]).reshape(self.N_, self.M_)

        output = np.zeros((2, self.N_, self.M_, 2))
        output[0, :, :, 0] = Ox
        output[0, :, :, 1] = Oy
        output[1, :, :, 0] = Cx
        output[1, :, :, 1] = Cy
        return output

    @property
    def marker_shape(self):
        return (self.N_, self.M_)
