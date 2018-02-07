
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 131 through 143 of the file called (`helper_functions.py`)[https://github.com/ikcGitHub/CarND-Vehicle-Detection/blob/master/helper_functions.py]).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Then I randomly picked an image from each classes and experienced with them.
I explored HOG feature extraction on the same image with different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).


| Parameters name  | First set | Second set | Third set |
| ------------- | ------------- | ------------- | ------------- |
| Orientation  | 9  | 9 | 9 |
| Pixels per cell  | 8  | 4 | 16 |
| Cells per block  | 2  | 2 | 2 |
| Visualize  | True  | True | True |
| Feature vector  | False  | False | False |


Here is an image showing the comparison in HOG feature extraction.


Then I tried spatial binning and color histogram in different color spaces


| Parameters name  | First set | Second set | Third set | Fourth set |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Color spaces  | 'RGb'  | 'RGB' | 'HLS' | 'YCrCb' |
| Spatial size  | (32, 32)  | (32, 32) | (32, 32) | (32, 32) |
| Histogram bins  | 32  | 32 | 32 | 32 |
| Histogram range  | (0, 256)  | (0, 256) | (0, 256) | (0, 256) |


Here is an image showing the comparison in color spaces feature extraction.


![alt text][image2]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and came up th following decision with explanation.
The documentation for `hog()` function can be found [here](http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog).

* For orientations, either 9 or 17 would be acceptable. I go with `orientation = 9`.
* For pixels_per_cell, smaller number will return more cells. For a 64 by 64 pixels picture, `pix per cell = 8` is good enough.
* For cells per block, either 2 or 4 would be good. I go with `cell_per_block = 2`.
* For visualize, I don't need the image return back in the following cells. Set `False` to it.
* For feature_vector, I need it to be a 1 demensional array for concatenating to other features. Therefore, set `True` to it.
* For multichannel, I am using `HLS` color space with `ALL` channels
___
__Final list__  
orient_final = 9  
pix_per_cell_final = 8  
cell_per_block_final = 2  
vis_final = False  
feature_vec_final = True  
multichannel_final = 'ALL'  


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the following features:
+ __Color features(spatial binning, color histogram)__  
color_space_final = 'HLS'  
spatial_size_final = (32, 32)  
hist_bins_final = 32  
hist_range_final = (0, 256)  
  
  
+ __HOG features__  
orient_final = 9  
pix_per_cell_final = 8  
cell_per_block_final = 2  
vis_final = False  
feature_vec_final = True  
multichannel_final = 'ALL'  
___
Here are the training steps:
+ Extract the specific features from the dataset
+ Normalie the extracted features vector
+ Define labels vector
+ Split up and randomize on both extracted features vector and labels vector
    - Return training dataset and test dataset
+ Create SVM
+ Fit the training dataset into SVM
+ Evaluate the accuracy of SVM
+ Predict the result with the test dataset
+ Done


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to set search window scale at `(64, 64)` and overlap at `(0.5, 0.5)` based on the experiment below. The code below is used to evaluate the performance of different combination of window scale and overlap value.  
From the final result, I come up the following conclusions:
* Larger overlap will result in more match windows
* Less overlap will result in less match window, nevertheless, not good for the following steps "add heat" and "filter".
* Smaller window will give you more match windows to analyze
* Bigger window will result less overlap, not good for the following steps "add heat" and "filter".

__Import functions__


```python
from helper_functions import slide_window, search_windows, draw_boxes
```

__Read in test image__


```python
image = mpimg.imread('./test_images/test1.jpg')
draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
image = image.astype(np.float32)/255
```

__Define Parameters__


```python
y_max = image.shape[0]
y_start_stop = [np.int(y_max/2), y_max] # Min and max in y to search in slide_window()

# First set
xy_window_1 = (64, 64)
xy_overlap_1 = (0.5, 0.5)

# Second set
xy_window_2 = (64, 64)
xy_overlap_2 = (0.2, 0.2)

# Third set
xy_window_3 = (64, 64)
xy_overlap_3 = (0.8, 0.8)

# Fourth set
xy_window_4 = (32, 32)
xy_overlap_4 = (0.5, 0.5)

# Fifth set
xy_window_5 = (96, 96)
xy_overlap_5 = (0.5, 0.5)
```

__Search windows__


```python
# First set
windows_1 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=xy_window_1, xy_overlap=xy_overlap_1)

hot_windows_1 = search_windows(image, windows_1, svc, X_scaler_final, color_space=color_space_final, 
                        spatial_size=spatial_size_final, hist_bins=hist_bins_final, hist_range = hist_range_final, 
                        orient=orient_final, pix_per_cell=pix_per_cell_final, 
                        cell_per_block=cell_per_block_final, 
                        hog_channel=multichannel_final, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

window_img_1 = draw_boxes(draw_image, hot_windows_1, color=(0, 0, 255), thick=6)

# Second set
windows_2 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=xy_window_2, xy_overlap=xy_overlap_2)

hot_windows_2 = search_windows(image, windows_2, svc, X_scaler_final, color_space=color_space_final, 
                        spatial_size=spatial_size_final, hist_bins=hist_bins_final, hist_range = hist_range_final, 
                        orient=orient_final, pix_per_cell=pix_per_cell_final, 
                        cell_per_block=cell_per_block_final, 
                        hog_channel=multichannel_final, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

window_img_2 = draw_boxes(draw_image, hot_windows_2, color=(0, 0, 255), thick=6)

# Third set
windows_3 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=xy_window_3, xy_overlap=xy_overlap_3)

hot_windows_3 = search_windows(image, windows_3, svc, X_scaler_final, color_space=color_space_final, 
                        spatial_size=spatial_size_final, hist_bins=hist_bins_final, hist_range = hist_range_final, 
                        orient=orient_final, pix_per_cell=pix_per_cell_final, 
                        cell_per_block=cell_per_block_final, 
                        hog_channel=multichannel_final, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

window_img_3 = draw_boxes(draw_image, hot_windows_3, color=(0, 0, 255), thick=6)

# Fourth set
windows_4 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=xy_window_4, xy_overlap=xy_overlap_4)

hot_windows_4 = search_windows(image, windows_4, svc, X_scaler_final, color_space=color_space_final, 
                        spatial_size=spatial_size_final, hist_bins=hist_bins_final, hist_range = hist_range_final, 
                        orient=orient_final, pix_per_cell=pix_per_cell_final, 
                        cell_per_block=cell_per_block_final, 
                        hog_channel=multichannel_final, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

window_img_4 = draw_boxes(draw_image, hot_windows_4, color=(0, 0, 255), thick=6)

# Fifth set
windows_5 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=xy_window_5, xy_overlap=xy_overlap_5)

hot_windows_5 = search_windows(image, windows_5, svc, X_scaler_final, color_space=color_space_final, 
                        spatial_size=spatial_size_final, hist_bins=hist_bins_final, hist_range = hist_range_final, 
                        orient=orient_final, pix_per_cell=pix_per_cell_final, 
                        cell_per_block=cell_per_block_final, 
                        hog_channel=multichannel_final, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       

window_img_5 = draw_boxes(draw_image, hot_windows_5, color=(0, 0, 255), thick=6)

```

__Plot images__


```python
# Plot the examples
f, axs = plt.subplots(2, 3, figsize=(36, 24))
axs = axs.ravel()

axs[0].imshow(window_img_1)
axs[0].set_title('xy_window=(64,64) xy_overlap=(0.5,0.5)', fontsize = 20)

axs[1].imshow(window_img_2)
axs[1].set_title('xy_window=(64,64) xy_overlap=(0.2,0.2)', fontsize = 20)

axs[2].imshow(window_img_3)
axs[2].set_title('xy_window=(64,64) xy_overlap=(0.8,0.8)', fontsize = 20)

axs[3].imshow(window_img_1)
axs[3].set_title('xy_window=(64,64) xy_overlap=(0.5,0.5)', fontsize = 20)

axs[4].imshow(window_img_4)
axs[4].set_title('xy_window=(32,32) xy_overlap=(0.5,0.5)', fontsize = 20)

axs[5].imshow(window_img_5)
axs[5].set_title('xy_window=(96,96) xy_overlap=(0.5,0.5)', fontsize = 20)

f.tight_layout()
plt.subplots_adjust(left=0.1, right=0.85, top=0.85, bottom=0.3)
```

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on one scale using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Those features extraction, classifying and window drawing were implemented in the function `find_cars()`. To enhance the efficiency of classifying, there are few code in `find_cars()` to extract HOG features just once for the entire region of interest in each full image / video frame. Please refer to the below images for your reference.

__Import functions__


```python
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import random

% matplotlib inline
```


```python
import pickle
from helper_functions import convert_color, find_cars
```

__Read in test images__


```python
# Images are in JPEG format
test_images = glob.glob('./test_images/*.jpg')

# Creat a list to store the file paths
test_images_all = []
for image in test_images:
    test_images_all.append(image)
```

__Define parameters__


```python
ystart = 400
ystop = 656
scale = 1.5
```

__Run the following code once a new classifier is created__

__Save a dictonary for future use__


```python
# # Create dictionary
# dict_vehicle_detection = {"svc": svc, 
#                           "X_scaler": X_scaler_final, 
#                           "orient": orient_final, 
#                           "pix_per_cell": pix_per_cell_final, 
#                           "cell_per_block": cell_per_block_final, 
#                           "feature_vec": feature_vec_final, 
#                           "spatial_size": spatial_size_final, 
#                           "hog_channel": multichannel_final, 
#                           "hist_bins": hist_bins_final, 
#                           "hist_range": hist_range_final, 
#                           "spatial_feat": spatial_feat, 
#                           "hist_feat": hist_feat, 
#                           "hog_feat": hog_feat,
#                           "ystart": ystart,
#                           "ystop": ystop,
#                           "scale": scale
#                          }

# # Save dictionary
# pickle.dump(dict_vehicle_detection, open('dict_vehicle_detection.p', 'wb'))
```

__Load data file__


```python
# Load dictionary
# dict_pickle = pickle.load(open('dict_vehicle_detection.p', 'rb'))
dict_pickle = pickle.load(open('HLS_ALL_HOG_9_8_2.p', 'rb'))

# Read data
svc = dict_pickle["svc"]
X_scaler = dict_pickle["X_scaler"]
orient = dict_pickle["orient"]
pix_per_cell = dict_pickle["pix_per_cell"]
cell_per_block = dict_pickle["cell_per_block"]
hog_channel = dict_pickle["hog_channel"]
spatial_size = dict_pickle["spatial_size"]
hist_bins = dict_pickle["hist_bins"]
hist_range = dict_pickle["hist_range"]
ystart = dict_pickle["ystart"]
ystop = dict_pickle["ystop"]
scale = dict_pickle["scale"]
```

__Find cars in image__


```python
# The actual number of features in your final feature vector will be the total number of block positions multiplied by the number of cells per block, times the number of orientations, or in the case shown above: 
# number of block positions = window size (pixels) / pixels per cell - 1
# Example: 7 = 96/12 -1
# 7×7×2×2×9=1764.

# Initialize an empty list
list_out_img = []
list_windows = []

for img_path in test_images_all:
    # Read image
    img = mpimg.imread(img_path)    
    
    # Scan for window scale (96, 96) and cells per step 8
    window_size = 96
    cell_per_step = 4
    pix_per_cell = 12 # Change pixe per cell to make the total number of block positions = 8
    # Find cars in the image
    out_img_1, list_windows_1 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, window_size, cell_per_step)
    # Scan for window scale (64, 64) and cells per step 4
    window_size = 64
    cell_per_step = 4
    pix_per_cell = 8 # Change pixe per cell to make the total number of block positions = 8
    # Find cars in the image
    out_img_2, list_windows_2 = find_cars(out_img_1, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, window_size, cell_per_step)
    # Append the output image to the list 
    list_out_img.append(out_img_2)
    # Append the matched windows to the list 
#     print(len(list_windows_1))
#     print(len(list_windows_2))
    if 0 == len(list_windows_1):
        hstack = list_windows_2
    elif 0 == len(list_windows_2):
        hstack = list_windows_1
    else:
        hstack = np.vstack((list_windows_1, list_windows_2))
    list_windows.append(hstack)
```


```python
# print(len(list_windows))
# print(len(list_windows[0]))
# print(len(list_windows[1]))
# print(len(list_windows[2]))
# print(len(list_windows[3]))
# print(len(list_windows[4]))
# print(len(list_windows[5]))
```

__Plot some example images__


```python
# Plot the examples
f, axs = plt.subplots(2, 3, figsize=(36, 24))
axs = axs.ravel()

for idx in range(len(list_out_img)):
    # Plot image
    axs[idx].imshow(list_out_img[idx])
    # Set title name
    title_name = test_images_all[idx].split('\\')[-1]
    axs[idx].set_title(title_name, fontsize = 20)

f.tight_layout()
plt.subplots_adjust(left=0.1, right=0.85, top=0.85, bottom=0.3)
```


![png](output_91_0.png)


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

__Import Libraries__


```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#import numpy as np
from helper_functions import process_image
```

__Process Video__  
Test on the given video project_video.mp4


```python
video_output = 'test_videos_output/project_video_output.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
# clip1 = VideoFileClip("project_video.mp4").subclip(0,3)
clip1 = VideoFileClip("project_video.mp4")
video_output_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time video_output_clip.write_videofile(video_output, audio=False)
```

    [MoviePy] >>>> Building video test_videos_output/project_video_output.mp4
    [MoviePy] Writing video test_videos_output/project_video_output.mp4
    

    100%|█████████▉| 1260/1261 [14:52<00:00,  1.42it/s]
    

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/project_video_output.mp4 
    
    CPU times: user 16min 48s, sys: 5.54 s, total: 16min 54s
    Wall time: 14min 53s
    

__Play Video Inline__


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video_output))
```





<video width="960" height="540" controls>
  <source src="test_videos_output/project_video_output.mp4">
</video>




#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

__Import librarires__


```python
from scipy.ndimage.measurements import label
```

__Import functions__


```python
from helper_functions import add_heat, apply_threshold, draw_labeled_bboxes
```


```python
# def add_heat(heatmap, bbox_list):
#     # Iterate through list of bboxes
#     for box in bbox_list:
#         # Add += 1 for all pixels inside each bbox
#         # Assuming each "box" takes the form ((x1, y1), (x2, y2))
#         heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

#     # Return updated heatmap
#     return heatmap# Iterate through list of bboxes
```


```python
# def apply_threshold(heatmap, threshold):
#     # Zero out pixels below the threshold
#     heatmap[heatmap <= threshold] = 0
#     # Return thresholded map
#     return heatmap
```


```python
# def draw_labeled_bboxes(img, labels):
#     # Iterate through all detected cars
#     for car_number in range(1, labels[1]+1):
#         # Find pixels with each car_number label value
#         nonzero = (labels[0] == car_number).nonzero()
#         # Identify x and y values of those pixels
#         nonzeroy = np.array(nonzero[0])
#         nonzerox = np.array(nonzero[1])
#         # Define a bounding box based on min/max x and y
#         bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
#         # Draw the box on the image
#         cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
#     # Return the image
#     return img
```

__Run functions__


```python
# list_out_img
# list_windows
```


```python
# Initialize an empty list
list_draw_img = []
list_heatmap = []

for idx in range(len(test_images_all)):
    # Read image
    image = mpimg.imread(test_images_all[idx])
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,list_windows[idx])

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    # Append the output image to the list 
    list_draw_img.append(draw_img)
    
    # Append the output heatmap to the list 
    list_heatmap.append(heatmap)
```

__Plot images__


```python
# Plot the draw images
f, axs = plt.subplots(6, 3, figsize=(36, 24))
axs = axs.ravel()

for idx in range(len(list_draw_img)):
    # Plot image
    axs[idx*3].imshow(list_out_img[idx])
    # Set title name
    # title_name = test_images_all[idx].split('\\')[-1]
    title_name = 'Car Positions'
    axs[idx*3].set_title(title_name, fontsize = 20)
    
    # Plot image
    axs[idx*3+1].imshow(list_draw_img[idx])
    # Set title name
    # title_name = test_images_all[idx].split('\\')[-1]
    title_name = 'Car Positions'
    axs[idx*3+1].set_title(title_name, fontsize = 20)

    # Plot image
    axs[idx*3+2].imshow(list_heatmap[idx], cmap='hot')
    # Set title name
    # title_name = test_images_all[idx].split('\\')[-1]
    title_name = 'Heatmap'
    axs[idx*3+2].set_title(title_name, fontsize = 20)

f.tight_layout()
plt.subplots_adjust(left=0.1, right=0.85, top=0.85, bottom=0.3)
```


![png](output_111_0.png)



```python
# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(draw_img)
# plt.title('Car Positions')
# plt.subplot(122)
# plt.imshow(heatmap, cmap='hot')
# plt.title('Heat Map')
# fig.tight_layout()
```

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 
