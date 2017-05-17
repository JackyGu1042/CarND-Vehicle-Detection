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
### Histogram of Oriented Gradients (HOG)

####1. Extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook .

I started by reading in all the `vehicle` and `non-vehicle` images. Beacause the cars is always appear in right side and far position, so I mainly use `vehicles/GTI_Right/image****.png` and `vehicles/GTI_Far/image****.png` as `vehicle` images, and use `non-vehicles/GTI/image****.png` as `non-vehicle` images. Moreover, the quantity is 1500 for both side. I used to try to use more pieces training images, but I found the linear SVM's performance is not good with bigger size of data.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). You can find the exmaple in the jupyter notebook file(or html file).

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I choose the parameter like below:
```python
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
```
I used to try many different combinations, finally I found the classifier need more features about HOG, so I choose `hog_channel = "ALL"` to get more feature.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using both bin_spatial(), color_hist() and get_hog_features(). And I found bin_spatial and hog feature are more important than color histograms.
```python
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

After combining the different feature vector, I use `StandardScaler()` method to normalize the training data.

```python
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
```
Finally, I use linear SVM to fit the data `svc = LinearSVC()`.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used to use 4 scales(32*32, 64*64, 96*96, 128*128) to capture the windows, like below code. But later, I found the overlap is more impatant than different scales type. So I decided to use only 2 scales(32*32, 96*96), and 32*32 scale scan in upon area and 96*96 scan wider range with high overlap rate. 

```python
windows_1 = slide_window(image, x_start_stop=[600, 1280], y_start_stop=[400, 528], 
                    xy_window=(32, 32), xy_overlap=(0.5, 0.5))

windows_2 = slide_window(image, x_start_stop=[600, 1280], y_start_stop=[400, 528], 
                    xy_window=(64, 64), xy_overlap=(0.8, 0.8))

windows_3 = slide_window(image, x_start_stop=[600, 1280], y_start_stop=[400, 720], 
                    xy_window=(96, 96), xy_overlap=(0.8, 0.8))

windows_4 = slide_window(image, x_start_stop=[600, 1280], y_start_stop=[400, 720], 
                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))
                    
windows =   windows_1 + windows_3  
```

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using `RGB` All-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. 

```python
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

