# **Advanced Lane Finding Project**

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

[undistortedChess]: ./examples/undistort_chess.png "Undistorted Chess"
[undistortedSample]: ./examples/undistort_sample.png "Undistorted Sample"
[thresholding1]: ./examples/thresholding1.png "Binary Example 1"
[thresholding2]: ./examples/thresholding2.png "Binary Example 2"
[perspective0]: ./examples/perspective0.png "Perspective"
[perspective1]: ./examples/perspective1.png "Perspective Example 1"
[perspective2]: ./examples/perspective2.png "Perspective Example 2"
[fit1]: ./examples/fit1.png "Fit Example 1"
[fit2]: ./examples/fit2.png "Fit Example 2"
[curvGen]: ./examples/curvGen.png "Curv generic formula"
[curv2ndop]: ./examples/curv2ndop.png "Curv formula for 2nd order poly"
[result1]: ./examples/result1.png "Result Example 1"
[result2]: ./examples/result2.png "Result Example 2"
[video1]: ./project_result.mp4 "Video"

My project includes the following files:
* calibration.py containing the script to calculate camera matrix and distortion coefficiens and save them to file
* perspective.py containing the script to calculate perspective transformation matrices and save them to file
* labutils.py containing routines to compute masks, gradients, to correct distortion, to warp perspective, to perform image analysis etc. 
* main.py containing the main classes, threshold routines, lane detection algorithm, and script to run pipeline
* project_result.mp4 containing resulting video produced by running pipeline on project_video.mp4
* writeup_report.md this file summarizing the results

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in file calibration.py (lines 34 to 59).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to one of calibrations images using the `cv2.undistort()` function and obtained these results: 

![Undistorted Chess][undistortedChess]

I precalculated camera matrix and distortion coefficients once and saved them to file. Later i loaded this data before video or single frame processing.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

For distortion correction i used cv2.undistort() function with previously calculated camera parameters (lines 24 to 32 of file labutils.py) 

Here is example of applying distortion correction:

![Undistorted Sample][undistortedSample]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of fixed range color threshoding, with adaptive thresholding of 'value' and 'saturation' channels of image in HSV color space.

Core idea of my adaptive thresholding routine, is an analysis of trapezoidal region of interest (ROI) which is mainly contains pixels corresponding to road surface. 
Most of these pixels with lower intensity corresponds to pavement, and few of them, that corresponds to markings, usually have higher intensity.
I build histogram of intencity of this ROI, and find first valley after maximum peak.
This can be done in many ways, i implemented two and chose one of them that performed better for project video. The code of intencity thresholding is contained in file main.py, lines 12 to 66.

Applying this routine to V and S channel of HSV converted image, i got two masks. 
I got another two for white and yellow colors: for white color i used RGB range [180,180,180]-[255,255,255], for yellow color i used range 15-25 on hue channel.
Then i took conjunction of V mask with white mask, and S mask with yellow mask, and disjunction of results.
I didn't use gradient thresholding because it gave no additional information on project video, but additional noise.
The code of whole thresholding step is contained in file main.py, lines 68 to 93.

Here are examples of my output for this step.  (note: this is not actually from one of the test images)

![thresholding1][thresholding1]
![thresholding2][thresholding2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I used single pair of direct and inverse perspective transformation matrices for whole video. 
I precalculated this matrices once and save them in a file to later use similary as for calibration parameters (lines 14 to 19 of file perspective.py). 
For this step i used one undistorted frame with straight lines from the project video. I chose centered trapezoid points as source points of transformation, and centered rectangle points as destination point.
I chose size of trapezoid such that perspective transform turns lane lines of frame to roughly parallel vertical lines.
I chose size of destination rectangle such that proportions in x and y directions were somewhat close to real, to decrease errors in calculations.
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
|  561, 470     | 320, 144      | 
|  719, 470     | 960, 144      |
| 1062, 690     | 960, 719      |
|  218, 690     | 320, 719      |

I verified that my perspective transform was working as expected by drawing the source trapezoid onto a test image and its warped counterpart with destination rectangle to verify that the lines appear parallel in the warped image.
Here is source trapezoid drawn with red color, and destination rectangle drawn with cyan.

![perspective0][perspective0]

Here are examples of perspective transformation applied to thresholded images from previous step:

![perspective1][perspective1]
![perspective2][perspective2]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used two modes of searching for pixels corresponding to lane lines. First - full search with sliding window - was used initially and when found lines didn't pass sanity checks.
Second - targeted search of nonzero pixels in some neighborhood of lines found on previous frame.

Full search starts by calculation of histogram of bottom half of frame, to find line centers as maximum values for left and right halfs of histogram.
Then sliding windows are placed at these centers at the bottom of frame and are moved up to top of frame, shifting centers to the mean x indices of nonzero pixels withing windows.
The code of this algorithm is contained in file main.py, lines 95 to 153.
The code of targeted search algorithm is contained in file main.py, lines 155 to 172.

I used numpy polyfit function to fit 2nd order polynomials to candidate pixels found by full or targeted search ( main.py, lines 188 to 199).  
Here are examples of found pixels (reds for left line, blues for right line) and fitted polynomials:

![fit1][fit1]
![fit2][fit2]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated vehicle position with respect to center of the lane as difference in pixels between lane center and frame center (asuming that camera was fixed on center plane of the car) multiplied by `meters per pixel` factor in x dimesion.

Code of vehicle position calculation is in line 387 of `main.py`

To calculate the radius of curvature, i applied general formula

![curvGen][curvGen]

to second order poynomial, which gives

![curv2ndop][curv2ndop]

where A and B are first and second polynomial coefficients respectively.

I applied this formula to fitted 2nd order polynomials of each line. 

Code of applying formula is in line 416 of `main.py`


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 569 through 583 in my code in `main.py` in the function `draw_lane()`.  Here is an example of my result on a frames from previous steps:

![result1][result1]
![result2][result2]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

As i mentioned above i used two modes of searching pixels corresponding to lane lines. If fitted polynomials were considered good, i used weighted (linear) averaging over 3 last fits to decrease jitter (main.py, lines 399 to 410).

Additionaly i implemented prediction mode, which is used when fitted polynomials didn't pass sanity cheks. Predictions were done by the same method as averaging.

My sanity checks includes checking that top and bottom widths of lane are reasonable: fit the frame, and do not deviate from average values for more then 10% (main.py, lines 207 to 249). 

The code for high-level algorithm of lines detection, including mode selection, sanity checks and state reset is contained in file main.py (lines 251 to 292)

I implemented Lane class to accumulate measurements from frame to frame, calclulate lane parameters and draw them on resulting frames (main.py, lines 314 to 668)

Here's a [link to my video result](./project_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Actually my goal was to implement pipeline robust enough to process all provided videos including harder challenge. But i faced few issues.

##### Thresholding 
Experiments with different videos showed that fixed range thresholding will not work with different cases. Shadows, light conditions, pavement changings, all that gives unacceptable results with fixed range thresholding.

Gradient thresholding sometimes can give aditional information but it is always noisy, what leads to problems with polynomial fitting after perspective transformation. Too wide line gives too much space for polynomial to fit.

Thats why i spent most of project time budget implementing different adaptive thresholding methods. But experimenting with these methods i found that i can perform very well one whole one video and most of others, but not for all special cases simultaniously.

Method that works well on simple cases + tunnel, will not work on pavement changings and so on.
Actually, i found that it is hard for me to even formulate the general optimisation task fitting all possible cases. 

And it looks pretty much like, for example, task of detection of cat on images.

So, i think, my best guess for going further here is to try CNN for lane-lines detection. 
 
##### Perspective transformation 
Static perspective transformation works rather satisfactorily with one video. Although when car moves from one pavement to another it jumps a little, and on these few frames static transformartion works really bad because of changed view angle of camera with respect to road plane.

But the same perspective transformation, calculated for project video, doesn't work with challenge video. 
And for harder challenge video, i guess, "road is a plane" assumption does not even holds.

So that all leads me to thoughts about dynamic perspective trasformation calculation. But here is sort of "chicken-egg" problem. To detect ROI for transformation estimation i'll need lane lines or, at least vanishing points. 

So here is the huge field of research for me. My first guess her is to look to "structure for motion" methods or similar methods of 3D scene estimation. Stereo vision looks promising as well, but does not meets project requirements - only one camera at my disposal.