import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os 
import labutils as lu

def find_chess_corners(gray, nx, ny, max_iterations=30, epsilon=0.001, search_win_size=(9,9)):
    '''
    Finds chessboard corners with refinement
    gray - source grayscaled image
    nx,ny - counts of corners in horizontal and vertical directions respectively
    max_iterations - maximum number of iterations for refinement
    epsilon - threshold to stop refinement
    search_win_size - tuple (width, height) - window size to search corners
    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, epsilon)
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, search_win_size, (-1,-1), criteria)
    return ret, corners

def calib(img_path, nx=9, ny=6, debug_show=False, debug_print=True):
    '''
    Calculates camera calibration matrix 
    nx,ny - expected inner corners on calibration images in x and y dimension respectively
    debug_show - when true, will show calibration images
    debug_print - when true, will print log in stdout
    '''
    if debug_print:
        print ('Starting calibration {} with nx={} ny={}'.format(img_path, nx,ny))
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    images = glob.glob(img_path)
    objpoints = [] 
    imgpoints = [] 
    for img_file in images:
        img = mpimg.imread(img_file)
        gray = lu.grayscale(img)
        ret, corners = find_chess_corners(gray,nx,ny)
        if debug_print:
            found = 0
            if ret:
                found = len(corners)
            print ('Processing {} Corners found: {}'.format(img_file, found))
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            if debug_show:
                img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
                win_name = os.path.basename(img_file)
                cv2.imshow(win_name,img)
                cv2.waitKey(0)
                cv2.destroyWindow(win_name)
    if debug_print:
        print ('Calibration completed')
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
imgs_path = '.\\camera_cal\\calibration*.jpg'
params_path = '.\\camera_cal\\calib_params.p'
ret, mtx, dist, rvecs, tvecs = calib(imgs_path)
lu.save({'mtx': mtx, 'dist': dist}, params_path)
print ('Params saved to: {}'.format(params_path))

calib_params = lu.load(params_path)

nx=9
ny=6
img = mpimg.imread('.\\camera_cal\\calibration3.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
ret, corners = find_chess_corners(gray,nx,ny)
img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
undist = lu.undistort(img, calib_params)
lu.plot_img_grid([img,undist],['Original','Undistorted'], cols=2, figsize=(7,3))

img = mpimg.imread('.\\test_images\\test1.jpg')
undist = lu.undistort(img, calib_params)
lu.plot_img_grid([img,undist],['Original','Undistorted'], cols=2, figsize=(7,3))
