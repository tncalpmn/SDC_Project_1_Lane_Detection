import numpy as np

LOWER_WHITE_THRES = [0,200,0]
HIGHER_WHITE_THRES = [255,255,255]

LOWER_YELLOW_THRES = [20, 80, 80]
HIGHER_YELLOW_THRES = [30, 255, 255]

GAUSSIAN_KERNEL = 3  # ORG 3
CANNY_LOW_THRES = 70  # ORG 70
CANNY_HIGH_THRES = 200  # ORG 200

LEFT_ROI = 0.095
RIGHT_ROI = 0.96
MIDDLE_LEFT_ROI = 0.47
MIDDLE_RIGHT_ROI = 0.55
MIDDLE_HEIGHT = 0.59

HOUGH_ROU = 1  # pixel Units
HOUGH_THETA = np.pi / 180  # Radian
HOUGH_THRESHOLD = 15  # Min Number of votes / intersection in a given grid
HOUGH_MIN_LINE_LENGT = 10  # After what distance should a line considered as a line
HOUGH_MAX_LINE_GAP = 40  # maximum distance between segments that is allowed to be connected into a single line.

MAX_LANE_HEIGHT = 0.62 # ORG 0.61

LEFT_LINE_COLOR = [255, 0, 0]
RIGHT_LINE_COLOR = [0, 0, 255]
LINE_THICKNESS = 2

LANE_THICKNESS = 8
LANE_COLOR = [0, 255, 0]

MAX_LEN_QUE = 10 # Take the average of last 10 Frames