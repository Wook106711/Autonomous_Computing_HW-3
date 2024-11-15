import numpy as np


def calc_depth_map(disp_left, k_left, t_left, t_right):

    ### START CODE HERE ###
    
    # Get the focal length from the K matrix
    f = k_left[0, 0]

    # Get the distance between the cameras from the t matrices (baseline)
    b = abs(t_left[1] - t_right[1])

    # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
    disp_left[disp_left == 0] = 0.1
    disp_left[disp_left == -1] = 0.1

    # Initialize the depth map to match the size of the disparity map
    depth_map = np.zeros_like(disp_left, dtype=np.float32)

    # Calculate the depths 
    depth_map = (f * b) / disp_left
    
    ### END CODE HERE ###
    
    return depth_map