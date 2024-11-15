import numpy as np
import cv2


def decompose_projection_matrix(p):
    
    ### START CODE HERE ###
    k, r, t = cv2.decomposeProjectionMatrix(p)[:3]

    t = t / t[3]
    
    ### END CODE HERE ###
    
    return k, r, t
