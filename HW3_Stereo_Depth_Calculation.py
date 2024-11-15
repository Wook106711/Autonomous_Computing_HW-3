import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches

import files_management
import comput_left_dispa_map
import decompose_project_matrix
import cal_depth_map


"""
arg : Image Load and Show, Projection Matrix Generation and Show
"""
# Read the stereo-pair of images (BGR -> RGB)
img_left = files_management.read_left_image()
img_right = files_management.read_right_image()

# Use matplotlib to display the two images (1행 2열 즉, 행방향 2개의 subplot 생성)
_, image_cells = plt.subplots(1, 2, figsize=(20, 20))
image_cells[0].imshow(img_left)
image_cells[0].set_title('left image')
image_cells[1].imshow(img_right)
image_cells[1].set_title('right image')
plt.show()


# Read the calibration (Projection Matrics 생성)
p_left, p_right = files_management.get_projection_matrices()

# Use regular numpy notation instead of scientific one 
np.set_printoptions(suppress=True)

print("p_left \n", p_left)
print("\np_right \n", p_right)



"""

"""
# Compute the disparity map using the fuction above
disp_left = comput_left_dispa_map.compute_left_disparity_map(img_left, img_right)

# Show the left disparity map
plt.figure(figsize=(10, 10))
plt.imshow(disp_left)
plt.show()


# Decompose each matrix
k_left, r_left, t_left = decompose_project_matrix.decompose_projection_matrix(p_left)
k_right, r_right, t_right = decompose_project_matrix.decompose_projection_matrix(p_right)

# Display the matrices
print("k_left \n", k_left)
print("\nr_left \n", r_left)
print("\nt_left \n", t_left)
print("\nk_right \n", k_right)
print("\nr_right \n", r_right)
print("\nt_right \n", t_right)


# Get the depth map by calling the above function
depth_map_left = cal_depth_map.calc_depth_map(disp_left, k_left, t_left, t_right)

# Display the depth map
plt.figure(figsize=(8, 8), dpi=100)
plt.imshow(depth_map_left, cmap='flag')
plt.show()