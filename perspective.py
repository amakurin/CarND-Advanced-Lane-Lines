import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import labutils as lu
import math

img_path = '.\\test_images\\straight_lines1.jpg' # frame-376 
params_path = '.\\camera_cal\\persp_params.p'

img = mpimg.imread(img_path) 
img = lu.undistort(img, lu.load('.\\camera_cal\\calib_params.p'))

src = lu.centered_trapezoid(img,bottom_ratio=.66, top_ratio=0.124, height = 250, bottom_crop = 30)
dst = lu.centered_rectangle(img, x_ratio=0.5, y_ratio=0.8)

M, Minv = lu.get_perspective_transform(np.float32(src), np.float32(dst))
params = {'direct': M, 'inverse': Minv, 'src': src, 'dst': dst}
lu.save(params, params_path)

img = cv2.polylines(img, [src], True, (255,0,0), thickness=2)
warped = lu.warp_perspective(img, params)
warped = cv2.polylines(warped, [dst], True, (0,255,255), thickness=3)

lu.plot_img_grid([img, warped], ['Initial undistorted', 'Warped'], rows=1, cols=2, figsize=(7,3))