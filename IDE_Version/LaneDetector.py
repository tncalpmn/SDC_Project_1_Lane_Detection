import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2
from moviepy.editor import VideoFileClip
from IDE_Version import Functions as func
from collections import deque
from IDE_Version import Constants as const

myQue = deque(maxlen=const.MAX_LEN_QUE)

def softenLanes(leftSlopeAndPoint, rightSlopeAndPoint):

    allLeft = [0,0,0,0]
    allRight = [0,0,0,0]

    myQue.append([leftSlopeAndPoint,rightSlopeAndPoint])

    for each in myQue:
        allLeft = [sum(x) for x in zip(allLeft, each[0])]
        allRight = [sum(x) for x in zip(allRight, each[1])]

    allLeftVar = np.array(allLeft) /  len(myQue)
    allRightVar = np.array(allRight) / len(myQue)

    softenedLeft = allLeftVar
    softenedRight = allRightVar

    return softenedLeft, softenedRight

def applyPipeline(image, stacked=False):

    imageWY = func.getOnlyWhiteAndYellows(image)
    grayScaleImage = func.grayscale(imageWY)
    blurredImage = func.gaussian_blur(grayScaleImage)
    cannyImage = func.canny(blurredImage)
    imageROI = func.region_of_interest(cannyImage)

    houghLines = func.hough_lines(imageROI)
    leftLines, rightLanes = func.seperateHoughLines(houghLines)

    if stacked: #To Generate Stacked Videos
        debug3D = np.stack((imageROI,) * 3, -1)
        leftAndRightLineMask, leftSlopeAndPoint, rightSlopeAndPoint = func.getLines(debug3D, leftLines, rightLanes, drawLeft=True, drawRight=True)
        linedImageROI =  func.weighted_img(debug3D, leftAndRightLineMask, α=0.8, β=1., γ=0.)
        lane_img= np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        softenedLeft, softenedRight = softenLanes(leftSlopeAndPoint, rightSlopeAndPoint)
        linedImageORG = func.drawLinesTo(image, lane_img ,softenedLeft, softenedRight)
        return np.hstack((linedImageORG, linedImageROI))
    else:
        leftAndRightLineMask, leftSlopeAndPoint, rightSlopeAndPoint = func.getLines(image, leftLines, rightLanes, drawLeft=False, drawRight=False)
        softenedLeft, softenedRight = softenLanes(leftSlopeAndPoint, rightSlopeAndPoint)
        linedImage = func.drawLinesTo(image, leftAndRightLineMask, softenedLeft, softenedRight)
        return linedImage

def saveTestImages(stacked = False):
    allImages = {}
    allImages["../test_images"] = os.listdir("../test_images/")
    allImages["../challenge_frames"] = os.listdir("../challenge_frames/")
    for eachKey in allImages:
        for imgName in allImages[eachKey]:
            if not imgName.startswith('.'):
                myQue.clear()
                path = eachKey + "/" + imgName
                image = mpimg.imread(path)
                finalImage = applyPipeline(image,stacked)
                cv2.imwrite('../test_images_output/' + imgName, cv2.cvtColor(finalImage, cv2.COLOR_RGB2BGR)) # OpenCV uses defaut BGR Scale, has to be convertet to RGB

def saveTestVideos():
    allVideos= os.listdir("../test_videos/")
    for vidName in allVideos:
        myQue.clear()
        outputFile = '../test_videos_output/' + vidName
        clip3 = VideoFileClip('../test_videos/' + vidName)
        challenge_clip = clip3.fl_image(applyPipeline)
        challenge_clip.write_videofile(outputFile, audio=False)

# Main
#saveTestImages()
#saveTestVideos()