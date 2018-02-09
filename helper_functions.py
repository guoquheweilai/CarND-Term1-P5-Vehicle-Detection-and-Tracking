import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.measurements import label
from collections import deque

n_frames = 6
heatmaps = deque(maxlen = n_frames)

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
    
# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes 
# for matched templates
def find_matches(img, template_list):
    # Make a copy of the image to draw on
    # Define an empty list to take bbox coords
    bbox_list = []
    # Define matching method
    # Other options include: cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCORR',
    #         'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
    method = cv2.TM_CCOEFF_NORMED
    # Iterate through template list
    for template in template_list:
        # Read in templates one by one
        temp = mpimg.imread(template)
        # Use cv2.matchTemplate() to search the image
        #     using whichever of the OpenCV search methods you prefer
        retval = cv2.matchTemplate(img, temp, method)
        # Use cv2.minMaxLoc() to extract the location of the best match
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(retval)
        # Determine bounding box corners for the match
        w, h = (temp.shape[1], temp.shape[0])
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = minLoc
        else:
            top_left = maxLoc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # Append bbox position to list
        bbox_list.append((top_left, bottom_right))
        # Return the list of bounding boxes
    return bbox_list
	
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

# Define a function to compute color histogram features  
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
# KEEP IN MIND IF YOU DECIDE TO USE THIS FUNCTION LATER
# IN YOUR PROJECT THAT IF YOU READ THE IMAGE WITH 
# cv2.imread() INSTEAD YOU START WITH BGR COLOR!
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    #features = img.ravel() # Remove this line!
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins = nbins, range = bins_range)
    ghist = np.histogram(img[:,:,1], bins = nbins, range = bins_range)
    bhist = np.histogram(img[:,:,2], bins = nbins, range = bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features, bin_centers, rhist, ghist, bhist

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features_bin_spatial_hist(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for img in imgs:
		# Read in each one by one
		image = mpimg.imread(img)
		# apply color conversion if other than 'RGB'
		if 'RGB' == cspace:
			features_image = np.copy(image)
		else:
			if 'HSV' == cspace:
				features_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
			elif 'LUV' == cspace:
				features_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
			elif 'HLS' == cspace:
				features_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
			elif 'YUV' == cspace:
				features_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
			elif 'YCrCb' == cspace:
				features_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
		# Apply bin_spatial() to get spatial color features
		spatial_feature = bin_spatial(features_image, size=spatial_size)
		# Apply color_hist() to get color histogram features
		# Remember to use [x] to read the specific return value
		color_feature = color_hist(features_image, nbins= hist_bins, bins_range=hist_range)[0]
		# Append the new feature vector to the features list
		features.append(np.concatenate((spatial_feature, color_feature)))
	# Return list of feature vectors
	return features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations = orient, pixels_per_cell = (pix_per_cell, pix_per_cell), cells_per_block = (cell_per_block, cell_per_block), visualise = vis, feature_vector = feature_vec, block_norm="L2-Hys")
        #features = [] # Remove this line
        #hog_image = img # Remove this line
        return features, hog_image
    else:      
        # Use skimage.hog() to get features only
        features = hog(img, orientations = orient, pixels_per_cell = (pix_per_cell, pix_per_cell), cells_per_block = (cell_per_block, cell_per_block), visualise = vis, feature_vector = feature_vec, block_norm="L2-Hys")
        #features = [] # Remove this line
        return features
	
# Define a function to extract features from a list of images
def extract_features_hog_channel(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for file in imgs:
		# Read in each one by one
		image = mpimg.imread(file)
		# apply color conversion if other than 'RGB'
		if 'RGB' == cspace:
			features_image = np.copy(image)
		else:
			if 'HSV' == cspace:
				features_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
			elif 'LUV' == cspace:
				features_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
			elif 'HLS' == cspace:
				features_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
			elif 'YUV' == cspace:
				features_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
			elif 'YCrCb' == cspace:
				features_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
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

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def extract_features_single_image(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
	#1) Define an empty list to receive features
	img_features = []
	#print("img is ", img)
	#2) Apply color conversion if other than 'RGB'
	if 'RGB' == color_space:
		features_image = np.copy(img)
	else:
		if 'HSV' == color_space:
			features_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif 'LUV' == color_space:
			features_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif 'HLS' == color_space:
			features_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif 'YUV' == color_space:
			features_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif 'YCrCb' == color_space:
			features_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)   
	#print('features_image is', features_image)
	#3) Compute spatial features if flag is set
	if spatial_feat == True:
		spatial_features = bin_spatial(features_image, size=spatial_size)
		#4) Append features to list
		# img_features.append(spatial_features.shape)
		#print('spatial_features size is ',len(spatial_features))
		#print("spatial_features is ", spatial_features)
	#5) Compute histogram features if flag is set
	if hist_feat == True:
		# Remember to use [x] to read the specific return value
		hist_features = color_hist(features_image, nbins=hist_bins, bins_range=hist_range)[0]
		#6) Append features to list
		# img_features.append(hist_features.shape)
		#print('hist_features size is ',len(hist_features))
		#print("hist_features is ", hist_features)
	#7) Compute HOG features if flag is set
	if hog_feat == True:
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(features_image.shape[2]):
				hog_features.extend(get_hog_features(features_image[:,:,channel], 
									orient, pix_per_cell, cell_per_block, 
									vis=False, feature_vec=True))      
		else:
			hog_features = get_hog_features(features_image[:,:,hog_channel], orient, 
						pix_per_cell, cell_per_block, vis=False, feature_vec=True)
		#8) Append features to list
		# img_features.append(hog_features)
		#print('hog_features size is ',len(hog_features))
		#print("hog_features is ", hog_features)
	#9) Return concatenated array of features
	img_features = np.hstack((spatial_features, hist_features, hog_features))
	return img_features
	#return np.concatenate(img_features)

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=(None, None), y_start_stop=(None, None), 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if None == x_start_stop[0]:
        x_start_stop[0] = 0
    if None == x_start_stop[1]:
        x_start_stop[1] = img.shape[1]
    if None == y_start_stop[0]:
        y_start_stop[0] = 0
    if None == y_start_stop[1]:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    span_x = x_start_stop[1] - x_start_stop[0]
    span_y = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    pixels_per_step_x = np.int(xy_window[0]*(1-xy_overlap[0]))
    pixels_per_step_y = np.int(xy_window[1]*(1-xy_overlap[1]))
    # Compute the number of windows in x/y
    pixels_first_block_x = np.int(xy_window[0]*xy_overlap[0])
    pixels_first_block_y = np.int(xy_window[1]*xy_overlap[1])
    n_w_x = np.int((span_x - pixels_first_block_x)/pixels_per_step_x)
    n_w_y = np.int((span_y - pixels_first_block_y)/pixels_per_step_y)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    for idx_w_y in range(n_w_y):
        for idx_w_x in range(n_w_x):
            # Calculate each window position
            coord_x_start = x_start_stop[0] + pixels_per_step_x * idx_w_x
            coord_x_end = xy_window[0] + coord_x_start
            coord_y_start = y_start_stop[0] + pixels_per_step_y * idx_w_y
            coord_y_end = xy_window[1] + coord_y_start
            # Append window position to list
            window_list.append(((coord_x_start, coord_y_start),(coord_x_end, coord_y_end)))
    # Return the list of windows
    return window_list

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using extract_features_single_image()
        features = extract_features_single_image(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range,
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# Define a function to convert image into different color spaces
def convert_color(img, color_space = 'HLS'):
    #2) Apply color conversion if other than 'RGB'
    if 'RGB' == color_space:
        features_image = np.copy(img)
    else:
        if 'HSV' == color_space:
            features_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif 'LUV' == color_space:
            features_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif 'HLS' == color_space:
            features_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif 'YUV' == color_space:
            features_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif 'YCrCb' == color_space:
            features_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)  
    return features_image
	
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, window_size, cell_per_step):
	# Initialize an empty list for matched window
	list_windows = []
	
	draw_img = np.copy(img)
	#img = img.astype(np.float32)/255
	img = img.astype(np.float32)

	img_tosearch = img[ystart:ystop,:,:]
	ctrans_tosearch = convert_color(img_tosearch, color_space='YCrCb')
	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
		
	ch1 = ctrans_tosearch[:,:,0]
	ch2 = ctrans_tosearch[:,:,1]
	ch3 = ctrans_tosearch[:,:,2]

	# Define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
	nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
	nfeat_per_block = orient*cell_per_block**2

	# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	window = window_size
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = cell_per_step  # Instead of overlap, define how many cells to step
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

	# Compute individual channel HOG features for the entire image
	if 'ALL' == hog_channel:
		hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
		hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
		hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
	else:
		hog_ch_fea = get_hog_features(ctrans_tosearch[:,:,hog_channel], orient, pix_per_cell, cell_per_block, feature_vec=False)
	
	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step
			# Extract HOG for this patch
			if 'ALL' == hog_channel:
				hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
				hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
				hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
				hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
			else:
				hog_features = hog_ch_fea[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 

			#print(hog_features.shape)

			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell

			# Extract the image patch
			subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

			# Get color features
			spatial_features = bin_spatial(subimg, size=spatial_size)
			hist_features = color_hist(subimg, nbins=hist_bins, bins_range=hist_range)[0]

			# print(spatial_features.shape)
			# print(hist_features.shape)

			# Scale features and make a prediction
			test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
			
			confidence_score = svc.decision_function(test_features)
			#print(confidence_score)
			
			if confidence_score >= 4:			
				#test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
				test_prediction = svc.predict(test_features)
			else:
				test_prediction = 0
				
			if test_prediction == 1:
				xbox_left = np.int(xleft*scale)
				ytop_draw = np.int(ytop*scale)
				win_draw = np.int(window*scale)
				cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
				list_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

	return draw_img, list_windows

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def plot3d(pixels, colors_rgb, axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation
	
def process_image(img):
	# Since image will be save in png format, scale up to 255
	draw_img = np.copy(img)
	img = img.astype(np.float32) * 255

	# Load dictionary
	path_classifier = './dict_vehicle_detection.p'
	#path_classifier = './classifiers/YCrCb_ALL_HOG_9_8_2.p'
	dict_pickle = pickle.load(open(path_classifier, 'rb'))
	# dict_pickle = pickle.load(open('dict_vehicle_detection.p', 'rb'))

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
	
	# Scan for window scale (96, 96) and cells per step 8
	window_size = 64
	cell_per_step = 6
	pix_per_cell = 8
	ystart = 420
	ystop = 580
	scale = 1.5
	# Run function
	out_img_1, list_windows_1 = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, window_size, cell_per_step)
	
	# Scan for window scale (64, 64) and cells per step 4
	window_size = 64
	cell_per_step = 6
	pix_per_cell = 8
	ystart = 400
	ystop = 540
	scale = 0.8
	# Run function
	out_img_2, list_windows_2 = find_cars(out_img_1, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, window_size, cell_per_step)
	
	# Create heatmap
	heat = np.zeros_like(out_img_2[:,:,0]).astype(np.float)

	# Combine windows list
	if 0 == len(list_windows_1):
		hstack = list_windows_2
	elif 0 == len(list_windows_2):
		hstack = list_windows_1
	else:
		hstack = np.vstack((list_windows_1, list_windows_2))
	# Add heat to each box in box list
	heat = add_heat(heat,hstack)

	# Add new heat map to the list
	heatmaps.append(heat)
	
	# Define threshold value
	threshold_filter = 6
	
	# Apply threshold to help remove false positives
	combined = sum(heatmaps)
	heat = apply_threshold(combined, threshold_filter)

	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)

	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(np.copy(draw_img), labels)
	
	return draw_img