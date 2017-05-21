**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_nocar.png
[image2]: ./output_images/RGB_shadow.PNG
[image3]: ./output_images/car_HOG.png
[image4]: ./output_images/slide_window.png
[image5]: ./output_images/teat1_output.png
[image6]: ./output_images/teat5_output.png
[image7]: ./output_images/teat3_output.png
[image8]: ./output_images/heat_map.png
[image9]: ./output_images/slide_window_detect.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook .

I started by reading in all the `vehicle` and `non-vehicle` images. Beacause the cars is always appear in right side and far position, so I mainly use `vehicles/GTI_Right/image****.png` and `vehicles/GTI_Far/image****.png` as `vehicle` images, and use `non-vehicles/GTI/image****.png` as `non-vehicle` images. Moreover, the quantity is 1500 for both side. I used to try to use more pieces training images, but I found the linear SVM's performance is not good with bigger size of data.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). You can find the exmaple in the jupyter notebook file(or html file).

Blew are two sample of training data:

![alt text][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I choose the parameter like below:
```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
```
I used to use `color_space = 'RGB`, although majority of video works well, but the tree shadow would always effect the result. So according to reviewer's suggestion , I use `YCrCb` to try, which could solve this problem.

![alt text][image2]

I used to try many different combinations, finally I found the classifier need more features about HOG, so I choose `hog_channel = 0` to get more feature.(Need change)

Blew is one sample for HOG image:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

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

Moreover, I use thresholding the decision function which helps to ensure high confidence predictions and reduce false positives.The decision function returns a confidence score based on how far away the sample is from the decision boundary. A higher decision function means a more confident prediction so by setting a threshold on it, you can ensure that you are only considering high confidence predictions as vehicle detections.

```python
prediction = clf.predict(test_features)
decision_value = clf.decision_function(test_features)
#
#7) If positive (prediction == 1) then save the window
if prediction == 1 and decision_value > 0.6:
    on_windows.append(window)
```
Finally, I use linear SVM to fit the data `svc = LinearSVC()`. And below is the result of SVM:

```
spatial_features: 3072
hist_features: 768
hog_features: 2352
Using: 12 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6192
6.97 Seconds to train SVC...
Test Accuracy of SVC =  0.9882
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

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

Below is the image with all sliding windows:

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using `YCrCb` 0-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. 

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

Below are some sample about test images' result:

![alt text][image5]

![alt text][image6]

![alt text][image7]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_result_20170517.mp4)

According to reviewer's advice, I add method to let detections be made around the same location in a sequence of multiple consecutive frames. I use simplest way, just store the heat maps of the 26 most recent frames in a list, take the sum of those heat maps, and set a threshold on the combined heat map to get the bounding boxes for each frame. This helps not only to filter out false positives, but it also makes the boxes to appear much smoother across frames.

```python
frame_len = len(frame_window)
hot_windows_frame = []

if frame_len == 0:
    frame_window.append(hot_windows)
    hot_windows_frame = frame_window[0]
    print("stage 1:",frame_len)
elif frame_len > 0 and frame_len <= frame_filter:
    frame_window.append(hot_windows)
    #hot_windows_frame = hot_windows
    for index in range(0,frame_len):
        hot_windows_frame = hot_windows_frame + frame_window[index] 
    print("stage 2:",frame_len)
elif frame_len > frame_filter:
    frame_window.append(hot_windows)
    del frame_window[0]
    #print(hot_windows)
    for index in range(0,frame_len):
        hot_windows_frame = hot_windows_frame + frame_window[index]
    #print(hot_windows_frame)
    print("stage 3:",frame_len)
```
Onemore vidoe is combine advance line detection and vehicle detection. Here's a [link to my video result](./project_video_result_20170517.mp4)

![alt text][image7]

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

```python
# Add heat to each box in box list
heat = add_heat(heat,box_list)
    
# Apply threshold to help remove false positives
heat = apply_threshold(heat,1)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(image), labels)
```
Below is the smaple of heat image:

![alt text][image9]

![alt text][image8]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In fact, I'm not very satisfied about this pipeline result, but because of deadline reason, I need submit this project. I think there are some points I need improve:
* Classifier is not good, I need to tunne some parameter for SVM or change another tool, like network.
* I think I need to improve the training data, current dataset is not very suitable for this project video.
* Enfficiency is very low, especially in sliding window search, I need optimize it.

