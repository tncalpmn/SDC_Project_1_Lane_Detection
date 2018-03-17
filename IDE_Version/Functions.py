import cv2
import math
from IDE_Version import Constants as const
import matplotlib.pyplot as plt
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def binaryScale(img): # Not Used
    imBW = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return imBW

def canny(img, low_threshold = const.CANNY_LOW_THRES, high_threshold = const.CANNY_HIGH_THRES):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size = const.GAUSSIAN_KERNEL):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, showROI=False):

    xDim = img.shape[1]
    yDim = img.shape[0]

    left = [int(xDim * const.LEFT_ROI), yDim - 1]
    right = [int(xDim * const.RIGHT_ROI), yDim - 1]
    middle_rigth = [int(xDim * const.MIDDLE_RIGHT_ROI), int(yDim * const.MIDDLE_HEIGHT)]
    middle_left = [int(xDim * const.MIDDLE_LEFT_ROI), int(yDim * const.MIDDLE_HEIGHT)]

    allCoordinates = [left, right, middle_rigth, middle_left]
    vertices = np.array([allCoordinates], dtype=np.int32)

    if showROI:
        showRegionOfInterest(img, allCoordinates)

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def converToHLS(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS);

def converToHSV(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV);

def showRegionOfInterest(image, vertices):

    print(image.shape)

    x, y = [], []
    for ver in vertices:
        x = x + [ver[0]]
        y = y + [ver[1]]

    x = x + [vertices[0][0]]
    y = y + [vertices[0][1]]

    plt.imshow(image)
    plt.plot(x, y, 'b--', lw=2)
    plt.title('Region of Interest')
    plt.show()

def hough_lines(img, rho = const.HOUGH_ROU, theta = const.HOUGH_THETA, threshold = const.HOUGH_THRESHOLD, min_line_len = const.HOUGH_MIN_LINE_LENGT, max_line_gap = const.HOUGH_MAX_LINE_GAP):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def seperateHoughLines(allHoughLines):
    leftLines, rightLines = [], []

    for line in allHoughLines:
        for x1, y1, x2, y2 in line:
            slope = ((y2 - y1) / (x2 - x1))
            if -0.90 < slope < -0.60:  # If slope is negative than Left -> Red
                leftLines.append(line)
            elif 0.30 < slope < 0.90:  # Slope positive Right -> Blue
                rightLines.append(line)

    return leftLines, rightLines

def getLines(image, leftLines, rightLines,drawLeft=True, drawRight=True):

    line_lane_img= np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    leftLinePoints = []
    rightLinePoints = []
    leftSlopeAndPoint = None
    rightSlopeAndPoint = None


    for line in leftLines:
        for x1, y1, x2, y2 in line:
            leftLinePoints.append((x1,y1))
            leftLinePoints.append((x2, y2))
            if drawLeft:
                cv2.line(line_lane_img, (x1, y1), (x2, y2), const.LEFT_LINE_COLOR, const.LANE_THICKNESS)

    for line in rightLines:
        for x1, y1, x2, y2 in line:
            rightLinePoints.append((x1,y1))
            rightLinePoints.append((x2, y2))
            if drawRight:
                cv2.line(line_lane_img, (x1, y1), (x2, y2), const.RIGHT_LINE_COLOR, const.LANE_THICKNESS)

    if leftLines:
        leftSlopeAndPoint = cv2.fitLine(np.array(leftLinePoints), cv2.DIST_L2, 0, 0.01, 0.01)
    if rightLines:
        rightSlopeAndPoint = cv2.fitLine(np.array(rightLinePoints), cv2.DIST_L2, 0, 0.01, 0.01)

    return line_lane_img, leftSlopeAndPoint, rightSlopeAndPoint

def drawLinesTo(image, line_lane_img, leftSlopeAndPoint,rightSlopeAndPoint, laneColor = const.LANE_COLOR, thickness = const.LANE_THICKNESS):

    maxHeightOfLine = int(image.shape[0] * const.MAX_LANE_HEIGHT)

    if leftSlopeAndPoint is not None:
        [vxx, vyy, xx, yy] = leftSlopeAndPoint
        xxLow = (((image.shape[0] - 1) - yy) * vxx / vyy) + xx
        xxHigh = ((maxHeightOfLine - yy) * vxx / vyy) + xx
        cv2.line(line_lane_img, ((int)(xxLow), (int)(image.shape[0] - 1)), ((int)(xxHigh), (int)(maxHeightOfLine)), laneColor, thickness)

    if rightSlopeAndPoint is not None:
        [vx, vy, x, y] = rightSlopeAndPoint
        xLow = (((image.shape[0] - 1) - y) * vx / vy) + x
        xHigh = ((maxHeightOfLine - y) * vx / vy) + x
        cv2.line(line_lane_img, ((int)(xLow), (int)(image.shape[0] - 1)), ((int)(xHigh), (int)(maxHeightOfLine)), laneColor, thickness)

    comboImage = weighted_img(image, line_lane_img, α=0.8, β=1., γ=0.)
    return comboImage

def getOnlyWhiteAndYellows(image, W = True, Y = True ):
    #cv2.inRange function derives the wanted colors between two upper and lower limit

    #White
    hlsImage = converToHLS(image);
    hsvImage = converToHSV(image);

    # HLS is better at finding white Images
    lowerWhite = np.array(const.LOWER_WHITE_THRES, dtype=np.uint8) ## HLS -> L brighness
    higherWhite = np.array(const.HIGHER_WHITE_THRES, dtype=np.uint8)
    whiteMask = cv2.inRange(hlsImage, lowerWhite, higherWhite)

    # HSV is better at finding Yellow Images
    lowerYellow = np.array(const.LOWER_YELLOW_THRES, dtype=np.uint8)
    higherYellow = np.array(const.HIGHER_YELLOW_THRES, dtype=np.uint8)
    yellowMask = cv2.inRange(hsvImage, lowerYellow, higherYellow)

    if W and Y:
        whiteYellowCombiMask =  cv2.bitwise_or(whiteMask, yellowMask)
        combiImage = cv2.bitwise_and(image, image, mask=whiteYellowCombiMask)
    elif Y:
        combiImage = cv2.bitwise_and(image, image, mask=yellowMask)
    elif W:
        combiImage = cv2.bitwise_and(image, image, mask=whiteMask)
    else:
        combiImage = image

    return combiImage

def createOutputDirectories():
    outputDir = ['test_images_output', 'test_videos_output']
    for eachOutDir in outputDir:
        if not os.path.exists(eachOutDir):
            os.makedirs(eachOutDir)

def showAndSave(image, imgName):
    plt.imshow(image)
    plt.title('Image ' + imgName + ' with Hough Lines combined')
    saveTo = 'test_images_output/' + imgName
    plt.axis('off')
    plt.savefig(saveTo, bbox_inches='tight')
    plt.show()

def showRegionOfInterest(image, coordinates):
    x, y = [], []
    for cor in coordinates:
        x = x + [cor[0]]
        y = y + [cor[1]]

    x = x + [coordinates[0][0]]
    y = y + [coordinates[0][1]]

    plt.imshow(image)
    plt.plot(x, y, 'b--', lw=2)
    plt.title('Region of Interest')
    plt.show()