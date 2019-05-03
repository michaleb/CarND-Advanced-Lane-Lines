## Advanced Lane Finding ##

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: Advanced-Lane-Finding/output_images/distorted_and_undistort_chess.png "Distorted & Undistorted Chessboard image"
[image2]: Advanced-Lane-Finding/output_images/undistorted_and_warped_test_image.jpg "Road Transformed"
[image3]: Advanced-Lane-Finding/output_images/binary_combined.png "Binary Example"
[image4]: Advanced-Lane-Finding/output_images/undistorted_and_warped_test_image.jpg "Warp Example"
[image5]: Advanced-Lane-Finding/output_images/fitted_poly.jpg "Fit Visual"
[image6]: Advanced-Lane-Finding/output_images/painted_lane.jpg "Output"
[video1]: Advanced-Lane-Finding/output_images/project_video.mp4 "Video"

              
---

### Camera Calibration

The calibration code is found in Section 1 of the Python code located at "./Advanced Lane Finding.py".  

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I  assumed the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

This is how I applied the distortion correction to an image:
![alt text][image2]

In section 1 of the notebook, the `cv2.calibrateCamera()` function outputs several values including the matrix (mtx) and distortion (dist) coefficients which I then used as input to the 'cv2.undistort()' function in section 2. The distortion coefficients correct errors produced in the image due to camera distance and alignment when the image was captured and the matrix includes the values for the focal lengths and optical centers (x,y directions) of the lens which are specific to the camera.


#### Creating Binary images

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in section 3 of code captioned "Convert to binary"). The image is converted to HSV colorspace and the saturation component used to detect the yellow and white lines. Next Sobel is used to calculate directional gradients which accentuate the high rates of change in both directions. Next the magnitude of directional gradients (Sobel-x and Sobel-y) is calculated. The rate of change of gradient with respect to the x-direction is calculated in the 'dir threshold()'function. All these functions were applied with threshold values derived through trial and error and the binary combined image produced is a logical combination of their outputs. Here's an example of my output for this section.  

![alt text][image3]

#### Perspective transform

The code for my perspective transform includes a function called `warp()`, which appears in section 5 of the code.  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points in the following manner:

```python
src = np.float32([[190,img.shape[0]],[1100,img.shape[0]],[550,480],[735,480]])
dst = np.float32([[190,img.shape[0]],[1100,img.shape[0]],[190,0],[1100,0]])
vertices = np.array([[(50,img.shape[0]),(1300,img.shape[0]),(600,350),(800,350)]], dtype=np.int32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 190, 720      | 
| 1100, 720     | 1100, 720     |
| 550, 480      | 190, 0        |
| 735, 480      | 1100, 0       |

I then verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### Identifying lane line pixels

The initial identification of the lane line pixels is done in section 7 - Pixel localization--- using the sliding window method. By creating a histogram to find peaks in the warped binary image, I used 75% of the frame instead of the 50% suggested to acquire more pixels representing the lane lines to compensate for 'noisy pixels' in the frame. The location of these peaks are taken as the base of the lane lines with the frame divided into equal size windows for left and right lines. The mean of non-zero pixel locations, above a threshold, bordered by the windows is calculated and then used to direct the 'sliding' of the window as the function iterates through the number of windows up the frame and this captures all changes in lane lines.

Since the lane lines vary slightly between frames section 11 - Focused search, used the previous frame's coefficients and the non-zero y-values in the current frame to generate a polynomial which serves as the starting point for searching for lane lines within a much tighter margin. The left and right (x,y) values are then used by function `fit_polynomial()` 

The output of section 7 and 11 are the left and right (x,y) indices of all non-zero pixels that make up the lane lines, this is fed into the function `fit_polynomial()` to generate the new frame's coefficients. The steps are in section 8 of the code. 

Here's an example of my output for this section.

![alt text][image5]

#### Radius of curvature and offset

The radius of curvature and offset are calculated in section 9 of the IPython notebook. By using the coefficients 'left_fit' and 'right_fit' each polynomial function was generated in pixel space, these x-values are then converted to meters and fed into the Python function np.polyfit() to produce coefficients representing 'real-world space'- meters. These coefficients are then used to calculate the radius of curvature. Two (2) new polynomial function are also generated and their x-values derived at the y-value closest to the car. These values are used to determine the offset of the camera with respect to the left and right fits by finding the middle of the fit and subtracting from the middle of the image frame.

#### Painted lane

I implemented this step in section 10 of the IPython notebook in the function `fill_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### Pipeline video output

Here's a [link to my video result](Advanced-Lane-Finding/output_images/project_video.mp4)

---

### Discussion

I used the HSV colorspace in this implementation as it is quite good in identifying the yellow lane line. However, I encountered issues when dealing with the shadowed areas and areas of high rates of change in pavement coloration on the road and so I had to utilize sanity checks to discard 'bad frames' and adjust the margin size of the `search_around_poly()` function.

I also used a region of interest mask to filter out 'noisy pixels' from the warped binary image which removed, for example, the change in gradient where the center concrete median meets the road surface of left lanes.

My pipeline fails when there is a reduction in the color contrast between the road surface and lane markings, for example in challenge video. It would also fail in the absence of road markings, for example in deep corners where lane markings may not be visible for several consecutive frames.

The pipeline could be improved by using another robust colorspace like LAB alongwith HSV for line detections. It would also improve if one lane marking could be used to determine another lane marking that is missing i.e. the lane markings that form the boundaries of a given lane are always equidistant from each other and of approximately the same curvature.



## Udacity's orginal README
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `output_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

