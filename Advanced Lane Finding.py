#!/usr/bin/env python

# ## Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 
# ---
# ## Section 1 - Get corner coordinates from distorted chessboard images

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        #img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

#cv2.destroyAllWindows()


# ## Section 2 - Camera calibration and undistort functions

def undistort_image(img):
    
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    # Undistorts an image given camera matrix and distortion coefficients, (mtx, dist)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst


# ## For display purposes only...

def show_before_and_after(img1, img2, img1_name, img2_name, cmap='gray'):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
    ax1.imshow(img1, cmap)
    ax1.set_title(img1_name, fontsize=20)
    ax2.imshow(img2, cmap)
    ax2.set_title(img2_name, fontsize=20)
    
    return

#Verifying correct camera calibration and undistortion using a calibration image

distorted_chessboard = mpimg.imread('../camera_cal/calibration1.jpg')
undistorted_chessboard = undistort_image(distorted_chessboard) 

show_before_and_after(distorted_chessboard, undistorted_chessboard,
                      img1_name = 'Distorted Chessboard', img2_name = 'Undistorted Chesboard')


# ##  Section 3 - Convert image to binary

def s_threshold(img, sobel_kernel=3, s_thresh=(170, 255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
    return s_binary

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    
    if orient == 'x':
        sobel = cv2.Sobel(S, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(S, cv2.CV_64F, 0, 1)
        
    # Absolute x or y derivatives to accentuate lines away from horizontal or vertical 
    abs_sobel = np.absolute(sobel) 
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Apply threshold
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    # 2) Take the gradient in x and y separately
    abs_sobelx = np.absolute(cv2.Sobel(S, cv2.CV_64F, 1, 0))
    abs_sobely = np.absolute(cv2.Sobel(S, cv2.CV_64F, 0, 1))
    # 3) Calculate the magnitude 
    abs_sobel_xy = (abs_sobelx**2 + abs_sobely**2)**0.5
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel_xy = np.uint8(255*abs_sobel_xy/np.max(abs_sobel_xy))
    # Apply threshold
    mag_binary = np.zeros_like(scaled_sobel_xy)
    mag_binary[(scaled_sobel_xy >= mag_thresh[0]) & (scaled_sobel_xy <= mag_thresh[1])] = 1
    
    return mag_binary

def dir_threshold(img, sobel_kernel=5, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #gray = cv2.cvtColor(hls, cv2.COLOR_HLS2GRAY)
    S = hls[:,:,2]
    
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(S, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(S, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(dir_grad)
    
    # 6) Return this mask as your binary_output image
    dir_binary[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
    
    return dir_binary


def apply_img_thresholds(img, ksize):
    
    # Apply each of the thresholding function
    s_binary = s_threshold(img, sobel_kernel=ksize, s_thresh=(150, 255))
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(20, 100))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.4))
    
    binary_combined = np.zeros_like(s_binary)
    binary_combined[(s_binary == 1) | ((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    return binary_combined


# ## Section 4 - Select pixels of interest in the binary image

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image


# ##  Section 5 - Warp image for perspective view

def warp(img, src, dst, img_size):
           
    #Compute the perspective transform, M, given source and destination points:
    M = cv2.getPerspectiveTransform(src, dst)

    #Warp an image using the perspective transform, M:
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
        
    return warped


# ## Section 6 - Unwarp perspective view

def unwarp(img, src, dst, img_size):
           
    #Compute the inverse perspective transform:
    Minv = cv2.getPerspectiveTransform(dst, src)

    #Warp an image using the perspective transform, M:
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_NEAREST)
        
    return unwarped


# ## Section 7 - Pixel localization using Sliding Window method

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom three quarters of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//4:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint 
    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 12
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base 
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
         
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
             
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty, out_img



# ## Section 8 - Fit curve function

def fit_polynomial(binary_warped, leftx, lefty, rightx, righty):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped) 
    
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    
    return left_fit, right_fit


# ## Section 9 - Calculate curvature and offset (in meters)

def measure_curve_and_offset_real(binary_warped, left_fit, right_fit, my, mx):
    '''
    Calculates the curvature of polynomial functions and car's offset from lane center in meters.
    '''
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    #Generate left and right x-values   
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] 
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Finding new poly coeff, using x and y values(in meters) to fit second order polynomial for both using `np.polyfit`
    # mx and my are the conversion factors (pixels to meters) for x and y values respectively
    left_fit_m = np.polyfit(my*ploty, mx*left_fitx, 2)
    right_fit_m = np.polyfit(my*ploty, mx*right_fitx, 2)
    
    # Calculates the maximum y-value (in meters), corresponding to the bottom of the image, closest to the car.
    y_eval = my*np.max(ploty)      
    
    #Generate left and right x-values at max y value, all in meters
    left_fitx_m = left_fit_m[0]*y_eval**2 + left_fit_m[1]*y_eval + left_fit_m[2] 
    right_fitx_m = right_fit_m[0]*y_eval**2 + right_fit_m[1]*y_eval + right_fit_m[2]
    
    #Finding the center of the generated lane lines at y-value closest to car    
    center_of_fit = (right_fitx_m + left_fitx_m)/2
    center_of_frame = 640*mx # 1/2 the x-dimension of image frame, converted to meters
    offset = round((center_of_fit - center_of_frame),2)    
        
    # Calculate the radius of curvature in meters for both lane lines 
    left_curverad = round(int((1 + (2*left_fit_m[0]*y_eval + left_fit_m[1])**2)**1.5) / np.absolute(2*left_fit_m[0]),2)
    right_curverad = round(int((1 + (2*right_fit_m[0]*y_eval + right_fit_m[1])**2)**1.5) / np.absolute(2*right_fit_m[0]),2)
    
        
    return left_curverad, right_curverad, offset


# ## Section 10 - Paint on lane

def fill_lane(frame, img_size, binary_warped, left_fit, right_fit, offset, RoC_L, RoC_R):
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0]) 
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = unwarp(color_warp, src, dst, img_size) 
    # Combine the result with the original image
    result = cv2.addWeighted(frame, 1, newwarp, 0.5, 0)
    
    if offset < 0:
        message = 'Car is left of center by '
    elif offset > 0:
        message = 'Car is right of center by '
    else:
        message = 'Car is center of lane '
    
    cv2.putText(result, 'Left Curvature = '+str(RoC_L)+'m', (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result, 'Right Curvature = '+str(RoC_R)+'m', (850,100),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result, str(message)+ str(offset)+'m',(10,150),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2,cv2.LINE_AA)

        
    return result


# ## Section 11 - Focused search

def search_around_poly(binary_warped, prev_left_fit, prev_right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 25

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
            
    left_lane_inds = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + 
                    prev_left_fit[2] - margin)) & (nonzerox < (prev_left_fit[0]*(nonzeroy**2) + 
                    prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin)))
    
    right_lane_inds = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + 
                    prev_right_fit[2] - margin)) & (nonzerox < (prev_right_fit[0]*(nonzeroy**2) + 
                    prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin)))
    
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    if (len(leftx) == 0 | len(lefty) == 0) | (len(rightx) == 0 | len(righty) == 0):
        left_fit = prev_left_fit
        right_fit = prev_right_fit
        
    else:
        # Fit new polynomials
        left_fit, right_fit = fit_polynomial(binary_warped, leftx, lefty, rightx, righty)
            
    return left_fit, right_fit


# ## Section 12 - Image pipeline
#
# Read in mage and get its dimensions
img = mpimg.imread('../test_images/straight_lines1.jpg')
img_size = (img.shape[1], img.shape[0])

#Applying the undistort function to a test image
frame = undistort_image(img)

src = np.float32([[190,img.shape[0]],[1100,img.shape[0]],[550,480],[735,480]])
dst = np.float32([[190,img.shape[0]],[1100,img.shape[0]],[190,0],[1100,0]])
vertices = np.array([[(50,img.shape[0]),(1300,img.shape[0]),(600,350),(800,350)]], dtype=np.int32)

ksize = 15 # Choose a larger odd number to smooth gradient measurements
# Meter per pixel conversion factor for both x and y dimensions for real world translations
mx = 3.7/(dst[1][0]-dst[0][0])
my = 30/(dst[0][1]-dst[2][1])

# Apply each of the thresholding function
binary_combined = apply_img_thresholds(frame, ksize)
warped_img = warp(frame, src, dst, img_size)     ## for display purposes only
ROI = region_of_interest(binary_combined, vertices)
binary_warped = warp(ROI, src, dst, img_size)
binary_combined_warped = warp(binary_combined, src, dst, img_size) ## for display purposes only

leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
left_fit, right_fit = fit_polynomial(binary_warped, leftx, lefty, rightx, righty)

RoC_L, RoC_R, offset = measure_curve_and_offset_real(binary_warped, left_fit, right_fit, my, mx)

result = fill_lane(frame, img_size, binary_warped, left_fit, right_fit, offset, RoC_L, RoC_R)

## Visualization ##

ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Colors in the left and right lane regions
out_img[lefty, leftx] = [255, 0, 0]
out_img[righty, rightx] = [0, 0, 255]

# Plots the left and right polynomials on the lane lines
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.imshow(out_img), plt.title('Fitted Polynomial')

show_before_and_after(out_img, result, img1_name='Identified lanes', img2_name='Filled Lane')
show_before_and_after(ROI, binary_warped, img1_name='Binary Combined [ROI]', img2_name='Binary Combined [ROI] Warped')
show_before_and_after(binary_combined, binary_combined_warped, img1_name='Binary Combined', img2_name='Binary Combined Warped')
#show_before_and_after(frame, warped_img, img1_name='Undistorted Image', img2_name='Warped Image')

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(frame, cmap='gray')
ax1.plot(src[0][0], src[0][1], 'o', markersize=20, markerfacecolor='red') # lower left
ax1.plot(src[1][0], src[1][1], 'o', markersize=20, markerfacecolor='red') # lower right
ax1.plot(src[2][0], src[2][1], 'o', markersize=20, markerfacecolor='red') # upper left
ax1.plot(src[3][0], src[3][1], 'o', markersize=20, markerfacecolor='red') # upper right

ax1.set_title('Undistorted Image')

ax2.imshow(warped_img, cmap='gray')
ax2.set_title('Warped Image', fontsize=20)
ax2.plot(dst[0][0], dst[0][1], 'o', markersize=20, markerfacecolor='red') # lower left
ax2.plot(dst[1][0], dst[1][1], 'o', markersize=20, markerfacecolor='red') # lower right
ax2.plot(dst[2][0], dst[2][1], 'o', markersize=20, markerfacecolor='red') # upper left
ax2.plot(dst[3][0], dst[3][1], 'o', markersize=20, markerfacecolor='red') # upper right
    


# ## Section 13 - Video pipeline
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def process_frame(img, img_size, src, dst, vertices, ksize):
    
    #Applying the undistort function 
    frame = undistort_image(img)
    # Apply each of the thresholding function
    binary_combined = apply_img_thresholds(frame, ksize)
    ROI = region_of_interest(binary_combined, vertices)
    binary_warped = warp(ROI, src, dst, img_size)
    
    return binary_warped, frame
    
def sliding_window(binary_warped):
    
    #Finding pixels coordinates for polynomial function and plotting lines
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit


'''
src = np.float32([[190,img.shape[0]],[1100,img.shape[0]],[550,480],[735,480]])
dst = np.float32([[190,img.shape[0]],[1100,img.shape[0]],[190,0],[1100,0]])
vertices = np.array([[(50,img.shape[0]),(1300,img.shape[0]),(600,350),(800,350)]], dtype=np.int32)

ksize = 15 # Choose a larger odd number to smooth gradient measurements
# Meter per pixel conversion factor for both x and y dimensions for real world translations
mx = 3.7/(dst[1][0]-dst[0][0])
my = 30/(dst[0][1]-dst[2][1])
'''

good_left_fit = []
good_right_fit = []
prev_left_fit = []
prev_right_fit = []
bad_frame = 0

def process_fps(img):
    global good_left_fit, good_right_fit, prev_left_fit, prev_right_fit, bad_frame
    
    fit = True   
    img_size = (img.shape[1], img.shape[0])
    binary_warped, frame = process_frame(img, img_size, src, dst, vertices, ksize)
    
    if len(prev_left_fit) == 0 | len(prev_right_fit) == 0:
        prev_left_fit, prev_right_fit = sliding_window(binary_warped)
        left_fit = prev_left_fit
        right_fit = prev_right_fit
                  
    else:
        left_fit, right_fit = search_around_poly(binary_warped, prev_left_fit, prev_right_fit)
        
        #Verify permissible range of distances between current left and right fit at y=0 
        if (right_fit[2] - left_fit[2]) < 700 or (right_fit[2] - left_fit[2]) > 1120:
            fit = False

        #Verify permissible range of distances between current left and right fit at y=720 
        left_fit_bottom = left_fit[0]*(720**2) + left_fit[1]*720 + left_fit[2]
        right_fit_bottom =  right_fit[0]*(720**2) + right_fit[1]*720 + right_fit[2]
        
        if (right_fit_bottom - left_fit_bottom) < 700 or (right_fit_bottom - left_fit_bottom) > 1120:
            fit = False
            
        #Compare slope [2Ay + B] of both fits at top of frame , y=0
        if left_fit[1] - right_fit[1] > 0.1:
            fit = False

        #Compare slope [2Ay + B] of both fits at bottom of frame , y=720
        left_slope_bottom = 2*left_fit[0]*720 + left_fit[1]
        right_slope_bottom = 2*right_fit[0]*720 + right_fit[1]

        if left_slope_bottom - right_slope_bottom > 0.1:
            fit = False

        if not fit:
            bad_frame += 1
            
            if bad_frame == 5:
                left_fit = prev_left_fit
                right_fit = prev_right_fit
                prev_left_fit = []
                prev_right_fit = []

            else:
                left_fit = prev_left_fit
                right_fit = prev_right_fit

    good_left_fit.append(left_fit)
    good_right_fit.append(right_fit)
    
    if fit:
        prev_left_fit = left_fit
        prev_right_fit = right_fit
        bad_frame = 0

    left_fit = np.mean(good_left_fit, axis=0)
    right_fit = np.mean(good_right_fit, axis=0)
        
    if len(good_left_fit) == 8:

        good_left_fit = []
        good_right_fit = []
        
        good_left_fit.append(left_fit)
        good_right_fit.append(right_fit)
        

    RoC_L, RoC_R, offset = measure_curve_and_offset_real(binary_warped, left_fit, right_fit, my, mx)
    result = fill_lane(frame, img_size, binary_warped, left_fit, right_fit, offset, RoC_L, RoC_R)
    
    return result


white_output = '../project_video.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("../project_video.mp4")
white_clip = clip1.fl_image(process_fps) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')

