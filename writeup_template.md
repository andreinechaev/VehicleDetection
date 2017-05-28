**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Histogram of Oriented Gradients (HOG)

####1. The code for this step is contained in the second code cell of the IPython notebook .  

I retrieved all `vehicle` and `non-vehicle` images from provided data set.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `LUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(32, 32)` and `cells_per_block=2`:


![alt text][image2]

####2. After some experiments I achieved 99.3% accuracy on the training set using parameters from the second block of the python notebook. Spatial binning dimensions made the most influence on the training. Increasing the value from `(8, 8)` to `(32, 32)`. The training rate increased from 92.5 to 99.3%. 
    Also, using all 3 channels for HOG improved the recognition rate and elimenated the problem with white cars. 
There is a small note. The learning rate is so good not only because of the best configs I used. The data set itself is really small. For the best accuracy and more precise results a bigger one should be used. 


####3.

I trained a linear SVM using `LinearSVC` the 5th block using default `hinge` loss functions with preprocessing data set with `StandartScaler` from `sklearn`.  


###Sliding Window Search

####1. I used two different techniques to find cars. 
- Sliding window. Slides window through a region with set offset. 
- HOG, sub-sampling. Instead of using offset for sliding through the region, it takes samples from the region with squares of size of HOG feature. 

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately, I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:
To reduce false positive I used `decision_function` with 1.2 threshold instead of `predict` functions that uses 1.0 by default.

![alt text][image4]
---

### Video Implementation

####1.
Here are [link to Sliding window search](./white.mp4)
and [link to Sub-sampling HOG search](./white2.mp4)


####2. I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Please take a look at the IPython notebook for more examples. 



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I noticed that despite a color schema I have extra artifacts at shadowed places. Perhaps, there should be more preprocessing. 
Also, the vehicle detection algorithm should follow the same car and track in and out of each during the stream. 
Different scale factors would help. It requires more experimentation but it should help. Although, it will require additional work with heatmaps. 

The system is still very slow. I'd like to know about optimization techniques. And really curious if GPU can significantly imporve the performance.
