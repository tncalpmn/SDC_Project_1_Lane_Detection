# **Project 1 - Finding Lane Lines on the Road**

## Project Definition

##### This project is from the first part of Udacity's Self Driving Car Nanodegree Program and the goal is to create a pipeline that detects the lane lines robustly in given images and videos.

---
### Project Folder
Short overview about which file contains what:
* challenge_frames : single frames from challenge video
* test_images : various test images
* test_videos : various test videos including challenge
* test_images_output : Output images after processed by my pipeline
* test_videos_output : Output videos after processed by my pipeline
* P1.ipynb - Jupyter Notebook for this project
* IDE_Version - Includes python files of this project for IDE use.
---

### Challenges
Various challenging circumstances in the visuals are:
* white or yellow lane lines
* straight or dashed lane lines
* shady roads due to trees
* patched asphalts
* due to camera position part of the car is also in image frame

---

### Aftermath - What I have used?

Here is a list of API functions that I have been using along this project.
* **cv2.cvtColor** -> color space converter

* **cv2.Canny** -> Used for edge detection in images

* **cv2.GaussianBlur** -> by given kernel size it blurs the image

* **cv2.fillPoly** -> apply colour to a part of image determined by the vertices of a polynomial

* **cv2.bitwise_and** -> applies pixel-wise & operation for two images

* **cv2.addWeighted** -> combine two images by given transparency thresholds

* **cv2.HoughLinesP** -> Probabilistic Hough lines finder, returns the start and end coordinates of a line in the given image (after processed by an edge detector, ex: Canny)

* **cv2.line** -> draw straight lines to given image

* **cv2.fitLine** -> applies linear regression to given set of points and returns the slope parameters (vx,vy) and a point (x,y) in fitted line.

* **cv2.inRange** -> selects pixels in the image only in given color code range

* **cv2.imwrite** -> saves the image  locally

* **deque** -> from collections library, double ended buffer/que, max length can be defined. If max length reached then delete entries with FIFO principle and append new entries

* **np.stack** -> Join a sequence of arrays along a new axis

* **np.hstack** -> join two arrays/Image in horizontal way (vstack is for vertical way)

* **np.zeros_like** -> returns a zero (0) (multi-dim.) array that has the same dimensions of given image

* **video.save_frame** -> saves the frame in a given second from video

---

### Reflection

### 1. Here is step by step, how I achieved to detect the lanes robustly

My pipeline consists of following steps:

1. Extract the yellow and white colours from given image/frame

2. Convert it to grayscale

3. Apply blur (Gaussian in my case) image in order to get better performance in the next step

4. Apply canny edge detection algorithm to actually get the image pixels where their value change drastically with respect to their neighbours (Canny algorithm actually applies its own Gaussian blur but I still applied before giving it to the function in order to get getter results)

5. Since I am interested in lines/lane in the image/frames, which are present only in front of the car, I simply ignore the rest of the image but that region where lanes are located by applying Region of Interest (ROI) mask. It is actually nothing but a  polygon that has four 2D corner points.

6. Since I had now canny edges in the region relevant to me, I apply Hough algorithm to find the lines given edges. This function returns the coordinates of starting & ending point coordinates of lines.

7. I have separated/ignored these lines according to their slope with following rules:
    * negative slope -> Left & -minThresLeft < slope < -maxTreshLeft (Red)
    * positive slope -> Right & minThresRight < slope < maxTreshRight (Blue)
    * Lines that does not fit into these conditions are simply ignored


8. Then I applied linear regression (cv2.fitLine) to the points that forms left and right lines, as a result i get the normalise vector (vx, vy) and a point in that line (x, y).

9. By using these four parameters I extrapolated the coordinates of the x coordinates, as y coordinates are simply provided by me: image.shape[0] and a location slightly below middle of the image.

10. So is per each frame a lane is detected. Although the lines seem nice in the images, in the videos they look a bit unsteady since each frame has its own line fit. To avoid that I saved the line information from last 10 frame in to a double sided buffer and first I took the average of these parameters then draw resulting lines on the original image. Thus, I achieved a smooth and robust lane detection.

![challangeStacked][allTogether]


Here is a [link](https://youtu.be/-QkNjW-tDrk) to check out how my pipeline performed on challenge video. Stacked version can be also found [here](https://youtu.be/0OmXC_0mDxI).

---
### 2. Potential shortcomings

All that said, this is a very naive approach to detect Lines. For every different circumstances parameters has to be changed. It might perform well in our test visuals but there is no guarantee that it will also behave the same in following circumstances:

* Dark roads
* Reflective wet roads due to weather conditions
* Sun coming directly to the camera
* When the vehicle turns, lines are not anymore straight

---

### 3. Possible improvements

* Rather than deterministic approach, more probabilistic approach could be used to fit lines in all kind of circumstances, for example with machine learning techniques.

* Lines could be fit non-linearly, especially when the vehicle turns.
* For sake of car to know what kind of lines are detected, for instance dashed or straight line, lanes could be drawn either in different colours or in dashed/straight way. So the car would know if it can overtake a car in front of it.

[stackedchallenge2]: ./test_images_output/Challenge_2_Stacked.jpg "ChallengeStacked"
[allTogether]: ./test_images_output/GitHub_Documentation/allTogether.jpg "ChallengeStacked"
