import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import labutils as lu
import math
from functools import partial
import copy
from enum import Enum
import collections as colls

def sobel_mag(img, sobel_kernel=3):
    sx = lu.sobel(img, dim='x', sobel_kernel=sobel_kernel)
    sy = lu.sobel(img, dim='y', sobel_kernel=sobel_kernel)
    mag = np.sqrt(sx*sx + sy*sy)
    return lu.scale(mag)

def sobel_dir(img, sobel_kernel=3):
    sx = lu.sobel(img, dim='x', sobel_kernel=sobel_kernel, absolute=True)
    sy = lu.sobel(img, dim='y', sobel_kernel=sobel_kernel, absolute=True)
    grad_direction = np.arctan2(sy, sx)
    return grad_direction

def equalize_light(img, clipLim = 1.0, gridSize = (16,16)):
    '''
    Applies adaptive histogram equalization to the light channel of an img
    `img` is an initial RGB image.
    Function converts img to Lab color scheme, applies CLAHE to light channel and converts back to RGB.
    `clipLim` and `gridSize` corresponds to opencv CLAHE params clipLimit and tileGridSize respectively.
    '''
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    L,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLim, tileGridSize=gridSize)
    cL = clahe.apply(L)
    lab = cv2.merge([cL,a,b])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    
def pipeline_v0(img, hsv_thresh = [[10,30],[56,255],[54,255]], 
                white_thresh=(250,255),
                grad_mag_thresh=(40,255),
                sobel_kernel=5):
    eq = equalize_light(img,clipLim = 12.0, gridSize = (4,4))
    eq = lu.grayscale(eq)
    white_mask = cv2.inRange(eq, white_thresh[0], white_thresh[1])

    gray = lu.grayscale(img)
    sx = lu.sobel(gray, sobel_kernel=sobel_kernel, absolute=True, scaling=True)
    sx_mask = cv2.inRange(sx, 20, 255)
    sy = lu.sobel(gray, dim='y', sobel_kernel=sobel_kernel, absolute=True, scaling=True)
    sy_mask = cv2.inRange(sy, 30, 255)
    grad_mag = sobel_mag(gray, sobel_kernel=sobel_kernel)
    mag_mask = cv2.inRange(grad_mag, grad_mag_thresh[0], grad_mag_thresh[1])

    grad_mask = (sx_mask | sy_mask) & mag_mask
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_thresh = np.array(hsv_thresh).T.reshape([2,3])
    hsv_mask = cv2.inRange(hsv, hsv_thresh[0], hsv_thresh[1])
    
    mask = white_mask | hsv_mask | grad_mask
    res = cv2.bitwise_and(img, img, mask = mask)
    return res, cv2.bitwise_and(img, img, mask = grad_mask), cv2.bitwise_and(img, img, mask = hsv_mask), cv2.bitwise_and(img, img, mask = white_mask)

def pipeline_v1(img, roi_mask, 
                hsv_thresh = [[10,30],[56,255],[54,255]], 
                white_thresh=(250,255),
                grad_mag_thresh=(40,255),
                sobel_kernel=5):
    eq = equalize_light(img,clipLim = 12.0, gridSize = (4,4))
    eq = lu.grayscale(eq)
    white_mask = cv2.inRange(eq, white_thresh[0], white_thresh[1])
    
    d = 20
    sigCol = 80
    sigSpa = 100
    img = cv2.bilateralFilter(img,d,sigCol,sigSpa)
    gray = lu.grayscale(img)
    
    sx = lu.sobel(gray, sobel_kernel=sobel_kernel, absolute=True, scaling=True)
    sx_mask = cv2.inRange(sx, 20, 255)
    sy = lu.sobel(gray, dim='y', sobel_kernel=sobel_kernel, absolute=True, scaling=True)
    sy_mask = cv2.inRange(sy, 30, 255)
    grad_mag = sobel_mag(gray, sobel_kernel=sobel_kernel)
    mag_mask = cv2.inRange(grad_mag, grad_mag_thresh[0], grad_mag_thresh[1])

    grad_mask = (sx_mask | sy_mask) & mag_mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    #grad_mask = cv2.morphologyEx(grad_mask, cv2.MORPH_OPEN, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    #grad_mask = cv2.morphologyEx(grad_mask, cv2.MORPH_CLOSE, kernel2)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_thresh = np.array(hsv_thresh).T.reshape([2,3])
    hsv_mask = cv2.inRange(hsv, hsv_thresh[0], hsv_thresh[1])
    
    mask = white_mask | grad_mask | hsv_mask 
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    res = cv2.bitwise_and(img, img, mask = mask)
    return res, cv2.bitwise_and(img, img, mask = grad_mask), cv2.bitwise_and(img, img, mask = white_mask), cv2.bitwise_and(img, img, mask = hsv_mask)

def find_perfect_valley(hist, high_threshold = 240):
    indices = np.arange(0,len(hist))
    low_threshold = np.asscalar(np.argmax(hist[:high_threshold]))
    kernel = [1, 0, -1]
    dhist = np.convolve(hist, kernel, 'same')
    dhist_sign = np.sign(dhist)
    dsign = np.convolve(dhist_sign, kernel, 'same')
    peaks = indices[(indices > low_threshold) & (indices < high_threshold)  & (dsign < 0)]
    valleys = indices[(indices > low_threshold) & (indices < high_threshold) & (dsign > 0)]
    #print ('half_mean:{}'.format(half_mean))
    #print ('peaks:{}'.format(peaks))
    #print ('valleys:{}'.format(valleys))
    x = hist[low_threshold:]
    y = x[x<np.mean(hist)]
    half_mean = np.mean(y) + np.std(y)
    peaks = peaks[hist[peaks] > half_mean]
    if len(peaks) > 0:
        low_threshold = np.asscalar(peaks[-1])
    valleys = valleys[(valleys > low_threshold) & (hist[valleys] < half_mean)]
    if len(valleys) > 0:
        low_threshold = np.asscalar(valleys[0])
    
    #fine tuning
    max_ind = low_threshold + np.argmax(hist[low_threshold:])
    path_to_max = hist[low_threshold:max_ind]
    offset = 0
    if len(path_to_max) > 1:
        addition = np.asscalar(np.argmin(path_to_max))
        if addition > 0:
            indices_scaled = indices/255
            lo_sum = np.sum(hist[low_threshold:low_threshold+addition]*indices_scaled[low_threshold:low_threshold+addition])
            hi_sum = np.sum(hist[low_threshold+addition:]*indices_scaled[low_threshold+addition:])
            offset = math.floor(addition*hi_sum/(lo_sum + hi_sum))
            #print ('addition={} lo_sum={} hi_sum={}'.format(addition, lo_sum, hi_sum))
    #print ('low_threshold={} offset={}'.format(low_threshold, offset))
    result = low_threshold + offset
    #plt.plot(hist,color='r')
    #plt.plot(dhist,color='b')
    #plt.plot(dsign*1000,color='g')
    #plt.xlim([0,len(dsign)])
    #plt.show() 
    return result
    
def find_perfect_valley_v2(hist):
    binned_size = 32
    binned = np.array([np.sum(hist[8*(i-1):8*i]) for i in range(1,binned_size+1)])
    binned = 100*binned/np.max(binned)
    bins = np.arange(binned_size)
    probs0 = np.zeros(binned_size)
    for i in bins[1:-1]:
        prob = 0
        if binned[i] > binned[i-1] or binned[i]>binned[i+1]:
            prob = 0 
        elif binned[i] < binned[i-1] and binned[i] == binned[i+1]:
            prob = .25 
        elif binned[i] == binned[i-1] and binned[i] < binned[i+1]:
            prob = .75 
        elif binned[i] < binned[i-1] and binned[i] < binned[i+1]:
            prob = 1. 
        else:
            prob = probs0[i-1] 
        probs0[i]=prob
    probs = np.zeros(binned_size)
    for i in bins[1:-1][::-1]:
        prob = 0
        if probs0[i] > 0:
            prsum = probs0[i-1]+probs0[i]+probs0[i+1]
            if prsum >= 1.:
                prob = 1.
        probs[i] = prob
    minima = bins[probs>0]
    maxbin = np.asscalar(np.argmax(binned))
    minima = minima[minima>maxbin]
    min_count = len(minima)
    if min_count==0:
        result = (maxbin+(binned_size-maxbin)//2)*8
    else:
        result = np.asscalar(minima[0])* 8
    #min_count = len(minima)
    ##print ('--')
    ##print(minima)
    ##print (maxbin)
    #if min_count==0:
    #    found_bin = maxbin + (binned_size-maxbin) // 2
    #elif min_count==1:
    #    found_bin = np.asscalar(minima[0])
    #else:
    #    fst_min = np.asscalar(minima[0])
    #    lst_min = np.asscalar(minima[-1])
    #    cucu = np.cumsum(binned[fst_min+1:lst_min+1][::-1])[::-1]
    #    #print(binned[minima])
    #    #print('first:{} others:{}'.format(binned[fst_min],cucu))
    #    found_bin = fst_min + len(cucu[cucu>binned[fst_min]])
    #result = found_bin * 8
    #print('found_bin:{} result:{}'.format(found_bin, result))
    return result
    
def thresh_intensity(int_chan, roi_mask):
    hist = cv2.calcHist([int_chan], [0], roi_mask, [256], [0,256])
    hist = np.reshape(hist,len(hist))
    low_threshold = find_perfect_valley(hist)
    #print ('low_threshold={}'.format(low_threshold))
    #lu.plot_hist(hist)
    int_mask = cv2.inRange(int_chan, low_threshold, 255)
    return int_mask

def thresh_intensity_v2(int_chan, roi_mask):
    hist = cv2.calcHist([int_chan], [0], roi_mask, [256], [0,256]).ravel()
    low_threshold = find_perfect_valley_v2(hist)
    #print ('low_threshold v2={}'.format(low_threshold))
    #lu.plot_hist(hist)
    int_mask = cv2.inRange(int_chan, low_threshold, 255)
    return int_mask

def thresh_grad_mag(img_chan, roi_mask, mean_ratio = .5, sobel_kernel = 5, gauss_kernel = 3):
    blurred = cv2.GaussianBlur(img_chan,(gauss_kernel,gauss_kernel),0)
    mag = sobel_mag(blurred, sobel_kernel=sobel_kernel)
    hist = cv2.calcHist([mag], [0], roi_mask, [256], [0,256])
    imax = np.argmax(hist)
    thresh = np.mean(hist)*mean_ratio    
    mag_thresh = next((i for i,v in enumerate(hist) if (i > imax) and (v<thresh)), 256)
    mag_mask = cv2.inRange(mag, mag_thresh, 255)
    return mag_mask
    
def pipeline_v2(img, roi_mask, debug=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    i_mask = thresh_intensity(hsv[:,:,2],roi_mask)
    ig_mask = thresh_grad_mag(hsv[:,:,2],roi_mask, mean_ratio=0.25, gauss_kernel=9)
    sg_mask = thresh_grad_mag(hsv[:,:,1],roi_mask)
    
    res_mask = i_mask | sg_mask | ig_mask
    if debug:
        return cv2.bitwise_and(img, img, mask = res_mask), cv2.bitwise_and(img, img, mask = i_mask), cv2.bitwise_and(img, img, mask = ig_mask), cv2.bitwise_and(img, img, mask = sg_mask)
    else:
        return cv2.bitwise_and(img, img, mask = res_mask)

def analysis(v, roi_mask):
    hist = cv2.calcHist([v], [0], roi_mask, [256], [0,256]).ravel()
    xxx =  find_perfect_valley_v2(hist.ravel())
    binned_size = 32
    binned = np.array([np.sum(hist[8*(i-1):8*i]) for i in range(1,binned_size+1)])
    binned = 100*binned/np.max(binned)
    bins = np.arange(binned_size)
    probs0 = np.zeros(binned_size)
    for i in bins[1:-1]:
        prob = 0
        if binned[i] > binned[i-1] or binned[i]>binned[i+1]:
            prob = 0 
        elif binned[i] < binned[i-1] and binned[i] == binned[i+1]:
            prob = .25 
        elif binned[i] == binned[i-1] and binned[i] < binned[i+1]:
            prob = .75 
        elif binned[i] < binned[i-1] and binned[i] < binned[i+1]:
            prob = 1. 
        else:
            prob = probs0[i-1] 
        probs0[i]=prob
    probs = np.zeros(binned_size)
    for i in bins[1:-1][::-1]:
        prob = 0
        if probs0[i] > 0:
            prsum = probs0[i-1]+probs0[i]+probs0[i+1]
            if prsum >= 1.:
                prob = 1.
        probs[i] = prob
    minima = bins[probs>0]
    maxbin = np.asscalar(np.argmax(binned))
    minima = minima[minima>maxbin]
    fst_min = minima[0]
    lst_min = minima[-1]
    cucu = np.cumsum(binned[fst_min+1:lst_min+1][::-1])[::-1]
    print ('--')
    print(minima)
    print(binned[minima])
    print('first:{} others:{}'.format(binned[fst_min],cucu))
    found = fst_min + len(cucu[cucu>binned[fst_min]])
    print('found:{}'.format(found))
    print('xxx:{} changed to {}'.format(xxx, found*8 ))
    xxx = np.asscalar(found*8)
    #plt.plot(hist,color='r')
    #plt.plot(hist2,color='b')
    #plt.xlim([0,len(hist2)])
    #plt.show() 

    plt.plot(binned,color='g')
    plt.plot(minima,binned[minima],'ro')
    plt.xlim([0,len(binned)])
    plt.show()
    return xxx#thre+thresh

def thresh_grad_x(int_chan, roi_mask, gauss_kernel=3, sobel_kernel=3, low_treshold = 20):
    blurred = cv2.GaussianBlur(int_chan,(gauss_kernel,gauss_kernel),0)
    sx = lu.sobel(blurred, dim='x', sobel_kernel=sobel_kernel, absolute=True, scaling=True)
    mean,std =cv2.meanStdDev(int_chan, mask = roi_mask)
    sx_mask = cv2.inRange(sx, low_treshold, 255) & cv2.inRange(int_chan, mean+std, 255)
    return sx_mask

ppp = '.\\test_images\\frame-532.jpg'#testc3 testch13 test1 straight_lines2.jpg frame-545 532 484 612 520
    
def pipeline_v3(img, roi_mask):
    res_mask = np.zeros(img.shape[:2],np.uint8)

    half_height = img.shape[0]//2
    cropped = img[half_height:,:,:]
    roi_mask = roi_mask[half_height:,:]
    hls = cv2.cvtColor(cropped, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    #h_mask = cv2.inRange(h, 0, 45)
    #v_mask = thresh_intensity(v, roi_mask)
    #s_mask = thresh_intensity(s, roi_mask)
    
    #lu.plot_img_grid([v,v_mask], rows=2, cols=1, figsize=(7,3))
    #lu.plot_img_grid([s,s_mask], rows=2, cols=1, figsize=(7,3))
    
    #vgrad_mask = thresh_grad_x(v,roi_mask)
    #sgrad_mask = thresh_grad_x(s,roi_mask)
    #lu.plot_img_grid([v,vgrad_mask, s, sgrad_mask], rows=2, cols=2, figsize=(7,3))
    #lu.plot_img_grid([cropped], rows=1, cols=1, figsize=(7,3))

    white_mask = cv2.inRange(cropped, np.array([180,180,180]), np.array([255,255,255]))
    #yellow_mask = cv2.inRange(cropped, np.array([120,190,0]), np.array([255,255,170]))

    yellow_mask = cv2.inRange(h, 15, 25)
    v_mask_v2 = thresh_intensity_v2(v, roi_mask)
    s_mask_v2 = thresh_intensity_v2(s, roi_mask)
    
    #lu.simple_thresh_tool(v)
    #lu.color_thresh_tool(img,cspace='rgb')#,cspace='hsv'(white_mask&v_mask_t)
    #lu.plot_img_grid([cropped, v_mask |(s_mask & h_mask), (v_mask_v2&white_mask)|(s_mask_v2& yellow_mask)], rows=3, cols=1, figsize=(7,3))
    res_mask[half_height:,:] =  (v_mask_v2 & white_mask) | (s_mask_v2 & yellow_mask)
    
    #res_mask[half_height:,:] =  (v_mask & white_mask) |(s_mask & h_mask)#| (vgrad_mask & white_mask) | (sgrad_mask & yellow_mask)

    
    return res_mask

def sliding_win_search(img_bin_warped, nwins=9, win_width=120, min_pix_recenter=50, hist_height_ratio=0.5):
    img_height = img_bin_warped.shape[0]
    win_height = img_bin_warped.shape[0]//nwins
    win_margin = win_width//2
    
    def new_window(win_index, x_center):
        btm = img_height - win_index * win_height
        top = btm - win_height
        left = x_center - win_margin
        right = x_center + win_margin
        return [top, btm, left, right]
        
    def which_within_window(y_inds, x_inds, window):
        [y_lo, y_hi, x_lo, x_hi] = window
        return ((y_inds >= y_lo) & (y_inds < y_hi) & (x_inds >= x_lo) & (x_inds < x_hi)).nonzero()[0]
        
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img_bin_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Calc hist of lower part of img 
    hist = np.sum(img_bin_warped[int(img_height * hist_height_ratio):,:], axis=0)
    midpoint = np.int(hist.shape[0]//2)
    left_center = np.argmax(hist[:midpoint])
    right_center = np.argmax(hist[midpoint:]) + midpoint
    left_lane_inds = []
    right_lane_inds = []
    windows = []
    for win in range(nwins):
        left_win = new_window(win, left_center)
        right_win = new_window(win, right_center)

        good_left_inds = which_within_window(nonzeroy, nonzerox, left_win)
        good_right_inds = which_within_window(nonzeroy, nonzerox, right_win)

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        windows.append([left_win, right_win])
        # If good pix > min_pix_recenter pixels, recenter next window on their mean position
        if len(good_left_inds) > min_pix_recenter:
            left_center = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > min_pix_recenter:        
            right_center = np.int(np.mean(nonzerox[good_right_inds]))
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    return [leftx, lefty], [rightx, righty], windows

def targeted_search(img_bin_warped, left_fit, right_fit, win_width=120):
    nonzero = img_bin_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = win_width//2
    left_x = np.polyval(left_fit, nonzeroy)
    left_lane_inds = ((nonzerox > (left_x - margin)) & (nonzerox < (left_x + margin))) 
    right_x = np.polyval(right_fit, nonzeroy)
    right_lane_inds = ((nonzerox > (right_x - margin)) & (nonzerox < (right_x + margin)))
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return [leftx, lefty], [rightx, righty]
    
def draw_sliding_windows(img, windows):
    def draw_window(img, win_bounds):
        [top, btm, left, right] = win_bounds
        cv2.rectangle(img, (left, top),(right, btm),(0,255,0), 2) 
        return img    
        
    for left,right in windows:
        draw_window(img, left)
        draw_window(img, right)
    return img    

def polyfit(xs, ys, x_ratio=1, y_ratio=1):
    result = None
    if (len(xs)>0) and (len(ys)>0):
        result = np.polyfit(ys*y_ratio, xs*x_ratio, 2)
    return result

def detect_lines(img_bin_warped, lane):
    def try_fit(lane, new_l_px, new_r_px):
        result = {'err': SearchErrors.Unknown}

        lpxlen = len(new_l_px[0])
        rpxlen = len(new_r_px[0])
        if (lpxlen>0) and (rpxlen>0):
            l_fit = polyfit(new_l_px[0], new_l_px[1])
            r_fit = polyfit(new_r_px[0], new_r_px[1])
            l_fitx = np.polyval(l_fit, lane.fity)
            r_fitx = np.polyval(r_fit, lane.fity)
            result = {'l_fit':l_fit, 'r_fit':r_fit, 'err': None}
            top_width = lane.calc_width_px(l_fitx, r_fitx, 0)
            btm_width = lane.calc_width_px(l_fitx, r_fitx, -1)
            if (top_width < 0) or (btm_width < 0) or (top_width > lane.frame_width) or (btm_width > lane.frame_width):
               print ('ERR: widths out of bounds lane.frame_width:', lane.frame_width, ' top_width:',top_width,' btm_width:',btm_width) 
               result = {'err': SearchErrors.WidthsOutOfBounds}
            else:   
                if lane.can_average():
                    top_width_avg = lane.avg_width_px(0)
                    if abs(top_width_avg - top_width) > top_width_avg*0.1:
                        print ('ERR: top_width_avg:',top_width_avg,' top_width:',top_width)
                        result = {'err': SearchErrors.TopWidthFarFromAvg}
                    btm_width_avg = lane.avg_width_px(-1)
                    if abs(btm_width_avg - btm_width) > btm_width_avg*0.1:
                        print ('ERR: btm_width_avg:',top_width_avg,' btm_width:',top_width) 
                        result = {'err': SearchErrors.BtmWidthFarFromAvg}
        else:
            print ('ERR: no pixels were found lpxlen:',lpxlen,' rpxlen:',lpxlen) 
            result = {'err': SearchErrors.NoPixelsFound}
        return result

    fit_result = None
    l_px = None
    r_px = None
    new_state = LaneStates.Undetected
    nxt_action = SearchActions.TargetSearch
    while new_state == LaneStates.Undetected:
        if (nxt_action == SearchActions.TargetSearch):
            nxt_action = SearchActions.FullSearch
            if lane.is_state_good():
                l_px, r_px = targeted_search(img_bin_warped, lane.l_fit_cur, lane.r_fit_cur)
                fit_result = try_fit(lane, l_px, r_px)
                if fit_result['err'] is None:
                    new_state = LaneStates.DetectedTarget
                    lane.update_success(fit_result['l_fit'], fit_result['r_fit'], l_px, r_px, new_state)
        if (nxt_action == SearchActions.FullSearch) and (new_state == LaneStates.Undetected):
            nxt_action = SearchActions.Prediction
            l_px, r_px, windows = sliding_win_search(img_bin_warped)
            fit_result = try_fit(lane, l_px, r_px)
            if fit_result['err'] is None:
                new_state = LaneStates.DetectedFull
                lane.update_success(fit_result['l_fit'], fit_result['r_fit'], l_px, r_px, new_state)
        if (nxt_action == SearchActions.Prediction) and (new_state == LaneStates.Undetected):
            nxt_action = SearchActions.Failing
            if lane.can_predict():
                if (lane.predictions <= lane.keep_n_fits) or (fit_result['err'] == SearchErrors.NoPixelsFound):
                    new_state = LaneStates.Predicted
                    lane.update_predict(l_px, r_px)
                else:
                    nxt_action = SearchActions.FullSearch
                    lane.update_reset()
        if (nxt_action == SearchActions.Failing) and (new_state == LaneStates.Undetected): 
            new_state = LaneStates.Failed
            lane.update_fail(l_px, r_px)
    return lane

class SearchActions(Enum):
    TargetSearch = 0
    FullSearch = 1
    Prediction = 2
    Failing = 3
    
class SearchErrors(Enum):
    Unknown = 0
    WidthsOutOfBounds = 1
    TopWidthFarFromAvg = 2
    BtmWidthFarFromAvg = 3
    NoPixelsFound = 4
    
class LaneStates(Enum):
    Undetected = 0
    Failed = 1
    Predicted = 2
    DetectedFull = 3
    DetectedTarget = 4

class Lane():
    def init_state(self, keep_n_fits):
        self.state = LaneStates.Undetected  
        self.predictions = 0
        # polynomial coefficients for n last fits
        self.l_fits = colls.deque(maxlen=keep_n_fits) 
        self.l_fitxs = colls.deque(maxlen=keep_n_fits) 

        self.r_fits = colls.deque(maxlen=keep_n_fits) 
        self.r_fitxs = colls.deque(maxlen=keep_n_fits) 

        self.l_fit_cur = None
        self.r_fit_cur = None

        self.l_fitx_cur = []
        self.r_fitx_cur = []
        
        self.l_px_cur = [[],[]]
        self.r_px_cur = [[],[]]
        pass
    
    def __init__(self, frame_size, xm_per_px, ym_per_px, keep_n_fits=3):
        self.keep_n_fits = keep_n_fits
        self.init_state(keep_n_fits)
        
        self.frame_height, self.frame_width = frame_size[:2]
        self.fity = np.linspace(0, self.frame_height-1, self.frame_height)
        
        self.xm_per_px = xm_per_px
        self.ym_per_px = ym_per_px
    
    def is_state_good(self):
        return self.state.value >  LaneStates.Failed.value
    
    def can_average(self):
        return (len(self.l_fits) > 0)
        
    def can_predict(self):
        return self.is_state_good() and self.can_average()

    def calc_width_px(self, l_fitx, r_fitx, index):
        return (r_fitx[index] - l_fitx[index])

    def cur_width_px(self, index):
        result = None
        if self.is_state_good():
            result = self.calc_width_px(self.l_fitx_cur, self.r_fitx_cur, index)
        return result
    
    def cur_vehicle_pos_px(self):
        return (self.frame_width/2 - (self.l_fitx_cur[-1] + self.cur_width_px(-1)/2))
    
    def avg_width_px(self, index):
        result = None
        if self.can_average():
            widths = [self.calc_width_px(l_fitx, r_fitx, index) for l_fitx, r_fitx in zip(self.l_fitxs, self.r_fitxs)]
            result = np.sum(widths)/len(widths)
        return result
    
    def avg_fit(self, fit_buf, fit=None):
        fits = list(fit_buf)
        if fit is not None:
            fits.append(fit)
        fitslen = len(fits)
        result = None
        if fitslen > 0:
            result = np.sum(fits*np.asarray([[i] for i in range(1,fitslen+1)]), axis = 0)*2/fitslen/(fitslen+1)
        return result
    
    def calc_curvature(self, fit, arg):
        return ((1 + (2 * fit[0] * arg + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
        
    def calc_curvature_m(self, px, arg):
        fit_m = np.polyfit(px[1] * self.ym_per_px, px[0] * self.xm_per_px, 2)
        return self.calc_curvature(fit_m, arg * self.ym_per_px)

    def l_curvature_px(self, arg):
        return self.calc_curvature(self.l_fit_cur, arg)
   
    def l_curvature_m(self, arg):
        return self.calc_curvature_m(self.l_px_cur, arg)
   
    def r_curvature_px(self, arg):
        return self.calc_curvature(self.r_fit_cur, arg)

    def r_curvature_m(self, arg):
        return self.calc_curvature_m(self.r_px_cur, arg)
        
    def append_fits(self, l_fit, r_fit):
        self.l_fits.append(l_fit)
        l_fitx = np.polyval(l_fit, self.fity)
        self.l_fitxs.append(l_fitx)
            
        self.r_fits.append(r_fit)
        r_fitx = np.polyval(r_fit, self.fity)
        self.r_fitxs.append(r_fitx)
        pass

    def set_cur_fits(self, l_fit, r_fit):
        self.l_fit_cur = l_fit
        self.r_fit_cur = r_fit
        self.l_fitx_cur = np.polyval(l_fit, self.fity)
        self.r_fitx_cur = np.polyval(r_fit, self.fity)
        pass
        
    def update_success(self, l_fit, r_fit, l_px, r_px, new_state=LaneStates.DetectedFull, average_fits=True):
        self.append_fits(l_fit, r_fit)
        if average_fits and self.can_average():
            l_fit = self.avg_fit(self.l_fits, l_fit)
            r_fit = self.avg_fit(self.r_fits, r_fit)
        self.set_cur_fits(l_fit, r_fit)    
        self.l_px_cur = l_px
        self.r_px_cur = r_px
    
        self.predictions = 0
        self.state = new_state
        return self
    
    def update_fail(self, l_px, r_px):
        self.l_fit_cur = None
        self.r_fit_cur = None
        self.l_px_cur = l_px
        self.r_px_cur = r_px
        self.state = LaneStates.Failed
        print ('ERR! Lane in Failed state. Predictions made:{}, loglen:{}'.format(self.predictions, len(self.l_fits)))
        return self

    def update_predict(self, l_px, r_px):
        if self.can_predict():
            l_fit = self.avg_fit(self.l_fits)
            r_fit = self.avg_fit(self.r_fits)
            self.append_fits(l_fit, r_fit)
            self.set_cur_fits(l_fit, r_fit)    
            self.predictions += 1
            self.l_px_cur = l_px
            self.r_px_cur = r_px
            self.state = LaneStates.Predicted
            return self
        else:
            print('WARN! Impossible to predict. Turning to Failed scenario.')
            return update_fail(self, l_px, r_px)

    def update_reset(self):
        print('WARN! Lane Reset called.')
        self.init_state(self.keep_n_fits)
        return self
        
    def draw_state(self, frame, pos, disk_r = 7, h_padding = 5):
        state_colors = [(255, 255, 255), (255, 0, 0), (255,164,84), (252, 221, 99), (149,230,87)]
        state_color = state_colors[self.state.value]
        pos = (pos[0]+disk_r, pos[1])
        cv2.circle(frame, pos, disk_r, state_color, -1) 
        pos = (pos[0]+disk_r+h_padding, pos[1]+disk_r)
        cv2.putText(frame, self.state.name, pos, cv2.FONT_HERSHEY_PLAIN, 1, state_color)
        pos = (pos[0] + 20*h_padding, pos[1])
        if self.predictions > 0:
            cv2.putText(frame, str(self.predictions), pos, cv2.FONT_HERSHEY_PLAIN, 1, state_color)
        return frame

    def draw_info(self, frame, pos):
        self.draw_state(frame, pos)
        if self.is_state_good():
            arg = np.median(self.fity)
            l_curv_m = round(self.l_curvature_m(arg),1)
            r_curv_m = round(self.r_curvature_m(arg),1)
            white = (255,255,255)
            font_size = 0.5
            v_padding = 30
            font = cv2.FONT_HERSHEY_DUPLEX
            pos = (pos[0], pos[1]+v_padding)
            cv2.putText(frame, 'L curv: {}m'.format(l_curv_m), pos, font, font_size, white)
            pos = (pos[0], pos[1]+v_padding)
            cv2.putText(frame, 'R curv: {}m'.format(r_curv_m), pos, font, font_size, white)
            pos = (pos[0], pos[1]+v_padding)
            cv2.putText(frame, 'Lane curv: {}m'.format(round((r_curv_m+l_curv_m)/2,1)), pos, font, font_size, white)
            pos = (pos[0], pos[1]+v_padding)
            vehicle_pos = round(self.cur_vehicle_pos_px() * self.xm_per_px, 1)
            cv2.putText(frame, 'Vehicle pos: {}m'.format(vehicle_pos), pos, font, font_size, white)
            #print (l_curv_m, r_curv_m)
        return frame

    def draw_lane(self, frame, persp_params):
        result = frame
        if self.is_state_good():
            # Create an image to draw the lines on
            zeros = np.zeros(frame.shape[:2]).astype(np.uint8)
            colored = np.dstack((zeros, zeros, zeros))
            
            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([self.l_fitx_cur, self.fity]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([self.r_fitx_cur, self.fity])))])
            pts = np.hstack((pts_left, pts_right))
            # Draw the lane onto the warped blank image
            cv2.fillPoly(colored, np.int_([pts]), (0,255, 0))
            
            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            unwarped = lu.warp_perspective(colored, persp_params, inverse=True)
            # Combine the result with the original image
            result = cv2.addWeighted(frame, 1, unwarped, 0.3, 0)
        return result

   
class Pipeline():
    def __init__(self, calib_params_path, persp_params_path,
                xm_per_px, ym_per_px,
                keep_n_fits=3,
                roi_mask=None):
        self.calib_params = lu.load(calib_params_path)
        self.persp_params = lu.load(persp_params_path)
        self.xm_per_px = xm_per_px
        self.ym_per_px = ym_per_px    
        self.roi_mask = roi_mask
        self.keep_n_fits = keep_n_fits
        self.lane = None
   
    def side_stack_imgs(self, imgs, side_y_ratio=0.25):
        result = imgs[0]
        total = len(imgs)
        if total>1:
            height,width = imgs[0].shape[:2]
            max_side_num = round((1-side_y_ratio)/side_y_ratio)
            if total < max_side_num+1:
                zeros = np.zeros_like(imgs[0])
                imgs.extend([zeros]*(max_side_num+1-total))
            elif total > max_side_num+1:
                imgs = imgs[:max_side_num+1]
            for i in range(1, max_side_num+1):
                imgs[i] = cv2.resize(imgs[i], (0,0), fx=side_y_ratio, fy=side_y_ratio)
                imgs[i][0,:,:] = 100
                imgs[i][-1,:,:] = 100
            side = np.vstack(imgs[1:])
            main_ratio = 1 - side_y_ratio
            main = cv2.resize(imgs[0], (0,0), fx=main_ratio, fy=main_ratio)
            result = np.hstack((main,side))
        return result
    
    def draw_finding_process(self, warped):
        result = np.dstack([warped]*3)
        result[self.lane.l_px_cur[1], self.lane.l_px_cur[0]] = [255, 0, 0]
        result[self.lane.r_px_cur[1], self.lane.r_px_cur[0]] = [0, 0, 255]
        ys = np.linspace(0, warped.shape[0]-1, warped.shape[0])
        if self.lane.is_state_good():
            cv2.polylines(result, [np.array(list(zip(self.lane.l_fitxs[-1], self.lane.fity)), np.int32)], False, [255, 255, 0], thickness=2)
            cv2.polylines(result, [np.array(list(zip(self.lane.r_fitxs[-1], self.lane.fity)), np.int32)], False, [255, 255, 0], thickness=2)

            cv2.polylines(result, [np.array(list(zip(self.lane.l_fitx_cur, self.lane.fity)), np.int32)], False, [255, 164, 84], thickness=2)
            cv2.polylines(result, [np.array(list(zip(self.lane.r_fitx_cur, self.lane.fity)), np.int32)], False, [255, 164, 84], thickness=2)
        return result
        
    def process_frame(self, frame):
        if self.lane is None:
            self.lane = Lane(frame.shape, self.xm_per_px, self.ym_per_px, self.keep_n_fits)
        if self.roi_mask is None:
            roi = lu.centered_trapezoid(frame, top_ratio = 0.2,height = 280, bottom_crop=35)
            self.roi_mask = lu.roi_mask(frame, roi)

        undistorted = lu.undistort(frame, self.calib_params)
        binarized = pipeline_v3(undistorted, self.roi_mask)
        warped = lu.warp_perspective(binarized, persp_params)
        new_lane = detect_lines(warped, self.lane)
        
        result = self.lane.draw_lane(undistorted, self.persp_params)
        result = self.side_stack_imgs([result, np.dstack([binarized]*3),
                                       self.draw_finding_process(warped)])
        result = self.lane.draw_info(result,(970, 380))        
        lu.plot_img_grid([result], rows=1, cols=1, figsize=(7,3))
        self.lane = new_lane
        return result     
        
    def process_video(self, src_path, tgt_path):
        self.lane = None
        lu.process_video(src_path, self.process_frame, tgt_path)
        pass
        
        

    
#ppp = '.\\test_images\\frame-133.jpg'#testc3 testch13 test1 straight_lines2.jpg frame-545 532 484 612 520
img = cv2.imread(ppp) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
calib_params_path='.\\camera_cal\\calib_params.p'
persp_params_path='.\\camera_cal\\persp_params.p'
calib_params = lu.load(calib_params_path)
persp_params = lu.load(persp_params_path)

pipeline = Pipeline(calib_params_path, persp_params_path, 3.7/615, 3/118)
#print(pipeline.left_line.keep_n_fits)  
result = pipeline.process_frame(img)
#pipeline.process_video('project_video.mp4', 'test40.mp4')
#pipeline.process_video('challenge_video.mp4', 'test40c.mp4')

print('---------------------------------------------------')

roi = lu.centered_trapezoid(img, top_ratio = 0.2,height = 280, bottom_crop=35)
r_mask = lu.roi_mask(img, roi)

img = lu.undistort(img, calib_params)
##mpimg.imsave('straight_lines2und.jpg', img)
#hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#h = hsv[:,:,0]
#s = hsv[:,:,1]
#v = hsv[:,:,2]
#
#res = pipeline_v3(img, r_mask)
#lu.plot_img_grid([img,res],cmaps=[None,'gray'], rows=1, cols=2, figsize=(7,3))

#bin = pipeline_v3(img, r_mask)
#bin = bin & r_mask
#warped = lu.warp_perspective(bin, persp_params)
##left_fitx, right_fitx = detect_lines(warped)
#ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
#lefts, rights, windows = sliding_win_search(warped)
#out = draw_sliding_windows(np.dstack((warped,warped,warped)).astype(np.uint8), windows)
#lu.plot_img_grid([img, out], rows=1, cols=2, figsize=(7,3))

#out_img,ploty, left_fitx,right_fitx = ttt(warped)

#out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
#out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

#result = draw(img,persp_params, ploty, left_fitx,right_fitx )
#lu.plot_img_grid([result,out_img], rows=1, cols=2, figsize=(7,3))

#plt.imshow(warped)
#plt.plot(left_fitx, ploty, color='cyan')
#plt.plot(right_fitx, ploty, color='cyan')
#plt.xlim(0, 1280)
#plt.ylim(720, 0)
#plt.show()
#




