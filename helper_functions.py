# Import libraries and functions
from skimage.feature import hog
import matplotlib.image as mpimg
import cv2
import numpy as np
import pickle
from scipy.ndimage.measurements import label
from collections import deque

n_frames = 12
heatmaps = deque(maxlen = n_frames)

# This method was duplicated from lesson materials
# Define a function to return some characteristics of the dataset 
def data_look(vehicle_list, non_vehicle_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_vehicles"] = len(vehicle_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_non_vehicles"] = len(non_vehicle_list)
    # Read in a test image, either car or notcar
    test_img = mpimg.imread(vehicle_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = test_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = test_img.dtype
    # Return data_dict
    return data_dict


# This method was duplicated from lesson materials
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Return two outputs if vis==True
    if True == vis:
		# Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Return one output in other cases
    else:
		# Use skimage.hog() to get features only
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# This method was duplicated from lesson materials
# Define a function to extract features from a list of images
def extract_hog_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img in imgs:
        # Read in each one by one
        image = np.copy(img)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, ystart, ystop, cspace, hog_channel, svc, X_scaler, orient, 
              pix_per_cell, cell_per_block, spatial_size, hist_bins, show_all_windows=False):
	list_windows =[]

	for scale in np.arange(1.0, 2.8, 0.2):
		windows = find_cars(img, ystart, ystop, scale, cspace, hog_channel, svc, None, orient, pix_per_cell, cell_per_block, None, None, False)[1]
		list_windows.append(windows)
		
	return list_windows
					
# This method was duplicated from lesson materials and adapted for different use
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, cspace, hog_channel, svc, X_scaler, orient, 
			  pix_per_cell, cell_per_block, spatial_size, hist_bins, show_all_windows=False):

	# array of windows where cars were detected
	windows = []

	draw_img = np.copy(img)
	img = img.astype(np.float32)/255

	img_tosearch = img[ystart:ystop,:,:]

	# apply color conversion if other than 'RGB'
	if cspace != 'RGB':
		if cspace == 'HSV':
			ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
		elif cspace == 'LUV':
			ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
		elif cspace == 'HLS':
			ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
		elif cspace == 'YUV':
			ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
		elif cspace == 'YCrCb':
			ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
	else: ctrans_tosearch = np.copy(image)   

	# rescale image if other than 1.0 scale
	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

	# select colorspace channel for HOG 
	if hog_channel == 'ALL':
		ch1 = ctrans_tosearch[:,:,0]
		ch2 = ctrans_tosearch[:,:,1]
		ch3 = ctrans_tosearch[:,:,2]
	else: 
		ch1 = ctrans_tosearch[:,:,hog_channel]

	# Define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell)+1  #-1
	nyblocks = (ch1.shape[0] // pix_per_cell)+1  #-1 
	nfeat_per_block = orient*cell_per_block**2
	# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	window = 64
	nblocks_per_window = (window // pix_per_cell)-1 
	cells_per_step = 2  # Instead of overlap, define how many cells to step
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step

	# Compute individual channel HOG features for the entire image
	hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)   
	if hog_channel == 'ALL':
		hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
		hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step
			# Extract HOG for this patch
			hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
			if hog_channel == 'ALL':
				hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
				hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
				hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
			else:
				hog_features = hog_feat1

			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell
			
			# Disable color feature extraction
			if 0:
				# Extract the image patch
				subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
			  
				# Get color features
				spatial_features = bin_spatial(subimg, size=spatial_size)
				hist_features = color_hist(subimg, nbins=hist_bins)

				# Scale features and make a prediction
				test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
				test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
				test_prediction = svc.predict(test_features)
			
			# test_prediction = svc.predict(hog_features.reshape(1, -1))
			# Use confidence score to determine the window should be included or not
			confidence_score = svc.decision_function(hog_features.reshape(1, -1))
			if confidence_score >= 0.5:			
				#test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
				test_prediction = 1
			else:
				test_prediction = 0
			
			if test_prediction == 1 or show_all_windows:
				xbox_left = np.int(xleft*scale)
				ytop_draw = np.int(ytop*scale)
				win_draw = np.int(window*scale)
				cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
				windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
				
	return draw_img, windows

# This method was duplicated from lesson materials
# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    random_color = False
    # Iterate through the bounding boxes
    for bbox in bboxes:
        if color == 'random' or random_color:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            random_color = True
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# This method was duplicated from lesson materials
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

# This method was duplicated from lesson materials
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
	
def draw_labeled_bboxes(img, labels):
	# Initialize an empty list for matched window
	list_windows = []

	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		list_windows.append(bbox)
		# Draw the box on the image
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
	# Return the image
	return img, list_windows

def process_test_image(img):
	# Load dictionary
	dict_pickle = pickle.load(open('dict_vehicle_detection.p', 'rb'))
	# Read data
	svc = dict_pickle["svc"]

	# Define parameters
	colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 11
	pix_per_cell = 16
	cell_per_block = 2
	hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
	ystart = 350
	ystop = 660
	
	# Search windows
	list_windows = search_windows(img, ystart, ystop, colorspace, hog_channel, svc, None, orient, pix_per_cell, cell_per_block, None, None, False)
	
	# Learned from stackflow (https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python)
	windows_flatten = [item for sublist in list_windows for item in sublist] 

	# Create heatmap
	heat_original = np.zeros_like(img[:,:,0]).astype(np.float)
	# Add heat to each box in box list
	heat = add_heat(heat_original,windows_flatten)
	
	# Apply threshold to help remove false positives
	heat_filtered = apply_threshold(np.copy(heat), 1)
	
	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat_filtered, 0, 255)
	
	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	
	# Draw bounding boxes on a copy of the image
	draw_img = draw_labeled_bboxes(np.copy(img), labels)[0]
	return draw_img
	
def process_image(img):
	# Load dictionary
	dict_pickle = pickle.load(open('dict_vehicle_detection.p', 'rb'))
	# Read data
	svc = dict_pickle["svc"]

	# Define parameters
	colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 11
	pix_per_cell = 16
	cell_per_block = 2
	hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
	ystart = 350
	ystop = 660
	
	# Search windows
	list_windows = search_windows(img, ystart, ystop, colorspace, hog_channel, svc, None, orient, pix_per_cell, cell_per_block, None, None, False)
	
	# Learned from stackflow (https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python)
	windows_flatten = [item for sublist in list_windows for item in sublist] 

	# Create heatmap
	heat_original = np.zeros_like(img[:,:,0]).astype(np.float)
	
	# Add heat to each box in box list
	heat = add_heat(heat_original,windows_flatten)
	
	# Add new heat map to the list
	heatmaps.append(np.copy(heat))
	
	# Define threshold value
	threshold_filter = 6
	
	# Apply threshold to help remove false positives
	combined = sum(heatmaps)
	heat_filtered = apply_threshold(combined, threshold_filter)
	
	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat_filtered, 0, 255)
	# heatmap_img = apply_threshold(heatmap_img, 1)
	
	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	
	# Draw bounding boxes on a copy of the image
	draw_img = draw_labeled_bboxes(np.copy(img), labels)[0]
	return draw_img