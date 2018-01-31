import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
    return rhist, ghist, bhist, bin_centers, hist_features
	
# Define a function to compute color histogram features  
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
# KEEP IN MIND IF YOU DECIDE TO USE THIS FUNCTION LATER
# IN YOUR PROJECT THAT IF YOU READ THE IMAGE WITH 
# cv2.imread() INSTEAD YOU START WITH BGR COLOR!
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if 'RGB' == color_space:
        features_image = np.copy(img)
    else:
        if 'HSV' == color_space:
            features_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif 'LUV' == color_space:
            features_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(features_image, size).ravel()
    #features = img.ravel() # Remove this line!
    # Return the feature vector
    return features
	
# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    test_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = test_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = test_img.dtype
    # Return data_dict
    return data_dict
	
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations = orient, pixels_per_cell = (pix_per_cell, pix_per_cell), cells_per_block = (cell_per_block, cell_per_block), visualise = vis, feature_vector = feature_vec)
        #features = [] # Remove this line
        #hog_image = img # Remove this line
        return features, hog_image
    else:      
        # Use skimage.hog() to get features only
        features = hog(img, orientations = orient, pixels_per_cell = (pix_per_cell, pix_per_cell), cells_per_block = (cell_per_block, cell_per_block), visualise = vis, feature_vector = feature_vec)
        #features = [] # Remove this line
        return features
