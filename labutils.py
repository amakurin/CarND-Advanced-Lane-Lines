import pickle
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import glob
import os 
from moviepy.editor import VideoFileClip, ImageSequenceClip

def grayscale(img):
    '''
    Convert img to grayscale
    '''
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def scale(img, max=255):
    '''
    Scales img 0 to max and conerts to uint8
    '''
    return np.uint8(max*img/np.max(img))
    
def undistort(img, camera_params):
    '''
    Performs distortion correction of img with respect to 
    camera_params - dictionary with mtx (camera matrix) and dist (distortion coefficients) keys 
    '''
    mtx = camera_params['mtx']
    dist = camera_params['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def sobel(img, dim = 'x', sobel_kernel=3, absolute=False, scaling=False):
    '''
    Calculates img gradient by applying sobel operator with respect to
    dim - if 'x' in x direction, 'y' in y direction 
    sobel_kernel - sobel operator kernel size
    absolute - if True takes absolute value of result
    scaling - if True scales result to 255 
    '''
    gray = img
    if len(img.shape)>2:
        gray = grayscale(img)
    s = cv2.Sobel(gray, cv2.CV_64F, dim == 'x', dim == 'y', ksize=sobel_kernel)    
    if absolute:
        s = np.absolute(s)
    if scaling:
        return scale(s)
    else:
        return s
    
def centered_trapezoid(img, top_ratio = 0.3, bottom_ratio = 0.95, height = 270, bottom_crop = 0):
    '''
    Calculates array of points for centered trapezoid aligned to bottom of img
    '''
    h, w = img.shape[:2]
    x_c = w // 2
    btm_y = h - bottom_crop
    top_y = h - height
    top_w = math.floor(w * top_ratio / 2)
    btm_w = math.floor(w * bottom_ratio / 2)
    return np.array([[[x_c - top_w, top_y],
                    [x_c + top_w, top_y],
                    [x_c + btm_w, btm_y],
                    [x_c - btm_w, btm_y]]], 
                    np.int32)

def centered_rectangle(img, x_ratio=0.9, y_ratio=1.):
    '''
    Calculates array of points for rectangle centered with img
    '''
    h, w = img.shape[:2]
    x_c = w // 2
    btm_y = h - 1
    top_y = h - math.floor(h * y_ratio)
    half_width = math.floor(w * x_ratio / 2)
    lft_x = x_c - half_width
    rgt_x = x_c + half_width
    return np.array([[[lft_x, top_y],
                    [rgt_x, top_y],
                    [rgt_x, btm_y],
                    [lft_x, btm_y]]], 
                    np.int32)
                    
def roi_mask(img, roi):
    '''
    Returns mask of img by points provided in roi argument 
    '''
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, roi, 255)
    return mask
    
def apply_mask(img, mask):
    '''
    Returns bitwise and if img and mask 
    '''
    return cv2.bitwise_and(img, img, mask = mask)

def get_perspective_transform(src, dst):
    '''
    Calculates forward and inverse perspective transformation matrices by points provided in src and dst
    '''
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv
    
def warp_perspective(img, transorm_params, inverse=False):
    '''
    Returns perspective transformation of img with respect to 
    transorm_params - map with 'direct' and 'inverse' keys - transformation matrices
    inverse - if true calculates inverse transform, direct transform - otherwise
    '''
    direction = 'direct'
    if inverse:
        direction = 'inverse'
    M = transorm_params[direction]
    img_size = img.shape[:2][::-1]
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def process_video(src_path, process_fn, tgt_path):
    '''
    Processes frames of video file on src_path, by process_fn and saves resulting video to file on tgt_path
    '''
    clip = VideoFileClip(src_path)
    white_clip = clip.fl_image(process_fn) 
    white_clip.write_videofile(tgt_path, audio=False)
    pass
    
def load(path):
    '''
    Loads serialized obj from file on path
    '''
    obj = pickle.load(open(path, 'rb'))
    return obj

def save(obj, path):
    '''
    Serializes obj and saves to file specified by path
    '''
    pickle.dump(obj, open(path, 'wb'))

##=====================================================================
## Analysis tools
##=====================================================================
def plot_hist(hist):
    '''
    Plots graph specified by hist and shows it in a window 
    '''
    plt.plot(hist)
    plt.xlim([0,len(hist)])
    plt.show()    
    
def plot_img_grid(images, titles=None, 
                    rows=1, cols=1, 
                    figid=None, figsize=(9, 4), 
                    hspace=0.0, wspace=0.0,
                    cmaps=None):
    '''
    Plots grid of specified images it in a window 
    '''
    cellscnt = rows*cols
    fig = plt.figure(figid, figsize)
    gs = gridspec.GridSpec(rows, cols, hspace=hspace, wspace=wspace)
    axs = [plt.subplot(gs[i]) for i in range(cellscnt)]
    imgslen = len(images)
    tlen = 0
    if titles is not None: 
        tlen = len(titles)
    cmlen = 0
    if cmaps is not None: 
        cmlen = len(cmaps)
    for i in range(cellscnt):
        if i < imgslen:
            img = images[i]
            if i < tlen: 
                axs[i].set_title(titles[i])
            if (i < cmlen) and cmaps[i] is not None:
                axs[i].imshow(img, cmap=cmaps[i])
            else:
                axs[i].imshow(img)
        axs[i].axis('off')
    gs.tight_layout(fig, pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.show()
    
def save_video_frames(src_path, tgt_path):
    '''
    Saves frames of video file on src_path to directory on tgt_path
    '''
    clip = VideoFileClip(src_path)
    frames = clip.iter_frames()
    src_filename = os.path.basename(src_path)
    fn = 'frame'
    ext = '.jpg'
    i = 0
    for frame in frames:
        print ('{}{}-{}{}'.format(tgt_path, fn, i, ext)) 
        mpimg.imsave('{}{}-{}{}'.format(tgt_path, fn, i, ext), frame)
        i = i + 1
    return frames

def do_nothing(x):
    '''
    Stub for ui tools
    '''
    pass
    
def simple_thresh_tool(img_chan, name='Simple_thresh', thresh_limits=(0,255)):
    '''
    Simple ui tool to threshold single channeled img_chan
    '''
    winname = name
    cv2.namedWindow(winname)
    cv2.createTrackbar('Min',winname,thresh_limits[0],thresh_limits[1],do_nothing)
    cv2.createTrackbar('Max',winname,thresh_limits[0],thresh_limits[1],do_nothing)
    while(1):
        min = cv2.getTrackbarPos('Min',winname)
        max = cv2.getTrackbarPos('Max',winname)
        mask = cv2.inRange(img_chan, min, max)
        res = cv2.bitwise_and(img_chan, img_chan, mask = mask)
        cv2.imshow(winname,res)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or cv2.getWindowProperty(winname, 0) < 0:
            break
    cv2.destroyWindow(winname)

def color_thresh_tool(rgb_img, name='Color_thresh', cspace='rgb', init=[[0,255],[0,255],[0,255]]):
    '''
    Three channel thresholding tool
    '''
    winname = name
    spaces_map = {'hls': cv2.COLOR_RGB2HLS, 
                'hsv': cv2.COLOR_RGB2HSV,
                'lab': cv2.COLOR_RGB2Lab,
                'rgb': None}
    converter = spaces_map[cspace]
    conv_img = np.copy(rgb_img)
    if converter is not None:
        conv_img = cv2.cvtColor(rgb_img, converter)
    
    cv2.namedWindow(winname)
    cv2.createTrackbar('Min0',winname,init[0][0],255,do_nothing)
    cv2.createTrackbar('Max0',winname,init[0][1],255,do_nothing)
    cv2.createTrackbar('Min1',winname,init[1][0],255,do_nothing)
    cv2.createTrackbar('Max1',winname,init[1][1],255,do_nothing)
    cv2.createTrackbar('Min2',winname,init[2][0],255,do_nothing)
    cv2.createTrackbar('Max2',winname,init[2][1],255,do_nothing)
    while(1):
        min0 = cv2.getTrackbarPos('Min0',winname)
        max0 = cv2.getTrackbarPos('Max0',winname)
        min1 = cv2.getTrackbarPos('Min1',winname)
        max1 = cv2.getTrackbarPos('Max1',winname)
        min2 = cv2.getTrackbarPos('Min2',winname)
        max2 = cv2.getTrackbarPos('Max2',winname)
        min = np.array([min0,min1,min2])
        max = np.array([max0,max1,max2])
        mask = cv2.inRange(conv_img, min, max)
        res = cv2.bitwise_and(rgb_img, rgb_img, mask = mask)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        res = res[300:-50,:]
        #res = cv2.resize(res, (0,0), fx=0.5, fy=0.5) 
        cv2.imshow(winname,res)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or cv2.getWindowProperty(winname, 0) < 0:
            break
    cv2.destroyWindow(winname)
