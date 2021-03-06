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

def find_perfect_valley(hist):
    '''
    Finds "optimal" minima of flattened histogram from 'hist'.  
    This version uses binning to smooth histogram, and funny method of valley finding without gradient calculation 
    (see ICIC Journal Volume 7, Number 10, October 2011 pp 5631-5644)
    '''
    # calculate binned hist
    binned_size = 32
    binned = np.array([np.sum(hist[8*(i-1):8*i]) for i in range(1,binned_size+1)])
    binned = 100*binned/np.max(binned)
    bins = np.arange(binned_size)
    # calculate valley probabilities estimations for each new bin
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
    # calculate final valley probabilities
    probs = np.zeros(binned_size)
    for i in bins[1:-1][::-1]:
        prob = 0
        if probs0[i] > 0:
            prsum = probs0[i-1]+probs0[i]+probs0[i+1]
            if prsum >= 1.:
                prob = 1.
        probs[i] = prob
    # all ones is valleys
    minima = bins[probs>0]
    # take first valley after maximum bin as result
    maxbin = np.asscalar(np.argmax(binned))
    minima = minima[minima>maxbin]
    min_count = len(minima)
    if min_count==0:
        result = (maxbin+(binned_size-maxbin)//2)*8
    else:
        result = np.asscalar(minima[0])* 8
    return result

def thresh_intensity(int_chan, roi_mask):
    '''
    Finds thresholding mask for intensity channel with respect to roi_mask by finding optimal valley of histogram 
    '''
    hist = cv2.calcHist([int_chan], [0], roi_mask, [256], [0,256]).ravel()
    low_threshold = find_perfect_valley(hist)
    int_mask = cv2.inRange(int_chan, low_threshold, 255)
    return int_mask
    
def binarize(img, roi_mask):
    '''
    Performs img binarization with respect to roi mask
    '''
    # create full sized empty mask
    res_mask = np.zeros(img.shape[:2],np.uint8)
    # create half height copies of img and roi mask to reduce computations
    half_height = img.shape[0]//2
    cropped = img[half_height:,:,:]
    roi_mask = roi_mask[half_height:,:]
    # convert to hsf space
    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    # color mask for white color
    white_mask = cv2.inRange(cropped, np.array([180,180,180]), np.array([255,255,255]))
    # color mask for yellow color
    yellow_mask = cv2.inRange(h, 15, 25)
    # value mask
    v_mask = thresh_intensity(v, roi_mask)
    # saturation mask
    s_mask = thresh_intensity(s, roi_mask)
    # result mask
    res_mask[half_height:,:] =  (v_mask & white_mask) | (s_mask & yellow_mask)
    return res_mask

def sliding_win_search(img_bin_warped, nwins=9, win_width=120, min_pix_recenter=50, hist_height_ratio=0.5):
    '''
    Performs sliding window search of lines on img_bin_warped by nwins windows of width win_width
    min_pix_recenter - minimum number of nonzero pixels in window to correct current center of windows
    hist_height_ratio - fraction of img height to calc histogram to find initial centers of windows
    '''
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
        # create current windows
        left_win = new_window(win, left_center)
        right_win = new_window(win, right_center)
        # findout indices of nonzero pixel withing windows
        good_left_inds = which_within_window(nonzeroy, nonzerox, left_win)
        good_right_inds = which_within_window(nonzeroy, nonzerox, right_win)
        # append found indices to accumulators
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
    '''
    Performs targeted search of lines on img_bin_warped, with respect to earlier found left_fit and right_fit
    win_width - width of windows
    '''
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
    '''
    Draws sliding windows on img
    '''
    def draw_window(img, win_bounds):
        [top, btm, left, right] = win_bounds
        cv2.rectangle(img, (left, top),(right, btm),(0,255,0), 2) 
        return img    
        
    for left,right in windows:
        draw_window(img, left)
        draw_window(img, right)
    return img    

def polyfit(xs, ys, x_ratio=1, y_ratio=1):
    '''
    Finds second order polynomial coefficients fitted with 
    xs - array of x indices
    ys - array of y indices
    x_ratio - scale coefficient in x direction
    y_ratio - scale coefficient in y direction
    '''
    result = None
    if (len(xs)>0) and (len(ys)>0):
        result = np.polyfit(ys*y_ratio, xs*x_ratio, 2)
    return result

def detect_lines(img_bin_warped, lane):
    '''
    Peforms lane lines detection
    img_bin_warped - binary, perspective wrapped image
    lane - current lane state
    '''
    def try_fit(lane, new_l_px, new_r_px):
        '''
        Tries to fit left and right polynomials with found pixels of lines. Peforms sanity cheks, predictions end lane state update
        lane - current lane state
        new_l_px - array of found pixels for the left line [x_indices_array, y_indices_array]
        new_r_px - array of found pixels for the right line [x_indices_array, y_indices_array]
        '''
        # init result with unknown error
        result = {'err': SearchErrors.Unknown}
        lpxlen = len(new_l_px[0])
        rpxlen = len(new_r_px[0])
        # if has pixels to fit - tries to fit
        if (lpxlen>0) and (rpxlen>0):
            l_fit = polyfit(new_l_px[0], new_l_px[1])
            r_fit = polyfit(new_r_px[0], new_r_px[1])
            l_fitx = np.polyval(l_fit, lane.fity)
            r_fitx = np.polyval(r_fit, lane.fity)
            result = {'l_fit':l_fit, 'r_fit':r_fit, 'err': None}
            # calculate lane top and bottom lane widths
            top_width = lane.calc_width_px(l_fitx, r_fitx, 0)
            btm_width = lane.calc_width_px(l_fitx, r_fitx, -1)
            # if widths are out of frame bounds - error
            if (top_width < 0) or (btm_width < 0) or (top_width > lane.frame_width) or (btm_width > lane.frame_width):
               print ('ERR: widths out of bounds lane.frame_width:', lane.frame_width, ' top_width:',top_width,' btm_width:',btm_width) 
               result = {'err': SearchErrors.WidthsOutOfBounds}
            else:   
                # if lane has info to average
                if lane.can_average():
                    # calculate average widths
                    top_width_avg = lane.avg_width_px(0)
                    # if new width is out of 10% of average - error
                    if abs(top_width_avg - top_width) > top_width_avg*0.1:
                        print ('ERR: top_width_avg:',top_width_avg,' top_width:',top_width)
                        result = {'err': SearchErrors.TopWidthFarFromAvg}
                    btm_width_avg = lane.avg_width_px(-1)
                    if abs(btm_width_avg - btm_width) > btm_width_avg*0.1:
                        print ('ERR: btm_width_avg:',top_width_avg,' btm_width:',top_width) 
                        result = {'err': SearchErrors.BtmWidthFarFromAvg}
        # if no pixels to fit - error
        else:
            print ('ERR: no pixels were found lpxlen:',lpxlen,' rpxlen:',lpxlen) 
            result = {'err': SearchErrors.NoPixelsFound}
        return result

    fit_result = None
    l_px = None
    r_px = None
    new_state = LaneStates.Undetected
    # start with targeted search
    nxt_action = SearchActions.TargetSearch
    # simple state machine
    while new_state == LaneStates.Undetected:
        if (nxt_action == SearchActions.TargetSearch):
            nxt_action = SearchActions.FullSearch
            # lane state is somewhat informative go on with targeted
            if lane.is_state_good():
                l_px, r_px = targeted_search(img_bin_warped, lane.l_fit_cur, lane.r_fit_cur)
                fit_result = try_fit(lane, l_px, r_px)
                # if no errors - finish algorithm
                if fit_result['err'] is None:
                    new_state = LaneStates.DetectedTarget
                    lane.update_success(fit_result['l_fit'], fit_result['r_fit'], l_px, r_px, new_state)
        if (nxt_action == SearchActions.FullSearch) and (new_state == LaneStates.Undetected):
            nxt_action = SearchActions.Prediction
            l_px, r_px, windows = sliding_win_search(img_bin_warped)
            fit_result = try_fit(lane, l_px, r_px)
            # if no errors - finish algorithm
            if fit_result['err'] is None:
                new_state = LaneStates.DetectedFull
                lane.update_success(fit_result['l_fit'], fit_result['r_fit'], l_px, r_px, new_state)
        if (nxt_action == SearchActions.Prediction) and (new_state == LaneStates.Undetected):
            nxt_action = SearchActions.Failing
            # if lane can predict, go on with prediction
            if lane.can_predict():
                # if no all log are predictions - predict and finish algorithm
                if (lane.predictions <= lane.keep_n_fits) or (fit_result['err'] == SearchErrors.NoPixelsFound):
                    new_state = LaneStates.Predicted
                    lane.update_predict(l_px, r_px)
                # if all log is predictions and some pixels were found on previous steps perform lane state reset and go on with full search
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
        # count of consequentive predictions 
        self.predictions = 0
        # polynomial coefficients for n last fits 
        self.l_fits = colls.deque(maxlen=keep_n_fits) 
        self.r_fits = colls.deque(maxlen=keep_n_fits) 
        # n last x indices for n last fits
        self.l_fitxs = colls.deque(maxlen=keep_n_fits) 
        self.r_fitxs = colls.deque(maxlen=keep_n_fits) 
        # current fits 
        self.l_fit_cur = None
        self.r_fit_cur = None
        # x indices for current fits
        self.l_fitx_cur = []
        self.r_fitx_cur = []
        # all pixels found for lines
        self.l_px_cur = [[],[]]
        self.r_px_cur = [[],[]]
        pass
    
    def __init__(self, frame_size, xm_per_px, ym_per_px, keep_n_fits=3):
        # max log sizes
        self.keep_n_fits = keep_n_fits
        self.init_state(keep_n_fits)
        # frame dims
        self.frame_height, self.frame_width = frame_size[:2]
        # y indices for fits
        self.fity = np.linspace(0, self.frame_height-1, self.frame_height)
        # scale coefficients meters per pixel
        self.xm_per_px = xm_per_px
        self.ym_per_px = ym_per_px
    
    def is_state_good(self):
        '''
        Checks if last measurement succeed
        '''
        return self.state.value >  LaneStates.Failed.value
    
    def can_average(self):
        '''
        Checks if has data to average
        '''
        return (len(self.l_fits) > 0)
        
    def can_predict(self):
        '''
        Checks if can predict next measurement
        '''
        return self.is_state_good() and self.can_average()

    def calc_width_px(self, l_fitx, r_fitx, index):
        '''
        Calculates lane width for specified y fitted index: 0 top frame, -1 bottom frame
        l_fitx - x indices fitted fot left line
        r_fitx - x indices fitted fot right line
        '''
        return (r_fitx[index] - l_fitx[index])

    def cur_width_px(self, index):
        '''
        Calculates lane width in pixels for specified y fitted index: 0 top frame, -1 bottom frame
        '''
        result = None
        if self.is_state_good():
            result = self.calc_width_px(self.l_fitx_cur, self.r_fitx_cur, index)
        return result
    
    def cur_vehicle_pos_px(self):
        '''
        Calculates vehicle position in pixels relative to center of lane 
        '''
        return (self.frame_width/2 - (self.l_fitx_cur[-1] + self.cur_width_px(-1)/2))
    
    def avg_width_px(self, index):
        '''
        Calculates average lane width in pixels for y fitted index: 0 top frame, -1 bottom frame  
        '''
        result = None
        if self.can_average():
            widths = [self.calc_width_px(l_fitx, r_fitx, index) for l_fitx, r_fitx in zip(self.l_fitxs, self.r_fitxs)]
            result = np.sum(widths)/len(widths)
        return result
    
    def avg_fit(self, fit_buf, fit=None):
        '''
        Calculates average polinomial fit with log specified by fit_buf 
        '''
        fits = list(fit_buf)
        if fit is not None:
            fits.append(fit)
        fitslen = len(fits)
        result = None
        if fitslen > 0:
            result = np.sum(fits*np.asarray([[i] for i in range(1,fitslen+1)]), axis = 0)*2/fitslen/(fitslen+1)
        return result
    
    def calc_curvature(self, fit, arg):
        '''
        Calculates radius of curvature of 'fit' in point specified by 'arg' 
        '''
        return ((1 + (2 * fit[0] * arg + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
        
    def calc_curvature_m(self, px, arg):
        '''
        Calculates radius of curvature of polynomial fitted in px, in point specified by 'arg' 
        px - array of pixels to fit polynomial [x_indices_array, y_indices_array]
        '''
        fit_m = np.polyfit(px[1] * self.ym_per_px, px[0] * self.xm_per_px, 2)
        return self.calc_curvature(fit_m, arg * self.ym_per_px)

    def l_curvature_px(self, arg):
        '''
        Calculates current left radius of curvature in pixels
        '''
        return self.calc_curvature(self.l_fit_cur, arg)
   
    def l_curvature_m(self, arg):
        '''
        Calculates current left radius of curvature in meters
        '''
        return self.calc_curvature_m(self.l_px_cur, arg)
   
    def r_curvature_px(self, arg):
        return self.calc_curvature(self.r_fit_cur, arg)

    def r_curvature_m(self, arg):
        return self.calc_curvature_m(self.r_px_cur, arg)
        
    def append_fits(self, l_fit, r_fit):
        '''
        Appends l_fit, r_fit and fitted x indices to log
        '''
        self.l_fits.append(l_fit)
        l_fitx = np.polyval(l_fit, self.fity)
        self.l_fitxs.append(l_fitx)
            
        self.r_fits.append(r_fit)
        r_fitx = np.polyval(r_fit, self.fity)
        self.r_fitxs.append(r_fitx)
        pass

    def set_cur_fits(self, l_fit, r_fit):
        '''
        Sets current fits and fitted x indices
        '''
        self.l_fit_cur = l_fit
        self.r_fit_cur = r_fit
        self.l_fitx_cur = np.polyval(l_fit, self.fity)
        self.r_fitx_cur = np.polyval(r_fit, self.fity)
        pass
        
    def update_success(self, l_fit, r_fit, l_px, r_px, new_state=LaneStates.DetectedFull, average_fits=True):
        '''
        Updates lane state after successful measurement
        '''
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
        '''
        Updates lane state after failed measurement
        '''
        self.l_fit_cur = None
        self.r_fit_cur = None
        self.l_px_cur = l_px
        self.r_px_cur = r_px
        self.state = LaneStates.Failed
        print ('ERR! Lane in Failed state. Predictions made:{}, loglen:{}'.format(self.predictions, len(self.l_fits)))
        return self

    def update_predict(self, l_px, r_px):
        '''
        Updates lane state by prediction
        '''
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
        '''
        Resets state
        '''
        print('WARN! Lane Reset called.')
        self.init_state(self.keep_n_fits)
        return self
        
    def draw_state(self, frame, pos):
        '''
        Draws state parameters on frame in specified pos
        '''
        state_colors = [(255, 255, 255), (255, 0, 0), (255,164,84), (252, 221, 99), (149,230,87)]
        state_color = state_colors[self.state.value]
        disk_r = 7
        h_padding = 5
        pos = (pos[0]+disk_r, pos[1])
        cv2.circle(frame, pos, disk_r, state_color, -1) 
        pos = (pos[0]+disk_r+h_padding, pos[1]+disk_r)
        cv2.putText(frame, self.state.name, pos, cv2.FONT_HERSHEY_PLAIN, 1, state_color)
        pos = (pos[0] + 20*h_padding, pos[1])
        if self.predictions > 0:
            cv2.putText(frame, str(self.predictions), pos, cv2.FONT_HERSHEY_PLAIN, 1, state_color)
        return frame

    def draw_info(self, frame, pos):
        '''
        Draws lane parameters on frame in specified pos
        '''
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
        '''
        Draws unwarped lane on frame
        '''
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
        '''
        Composes imgs to main main area (first img) and side bar (all others)
        '''
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
        '''
        Draws warped frame, found pixels, measured fits, averaged fits
        '''
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
        '''
        Main frame processing pipeline
        '''
        if self.lane is None:
            self.lane = Lane(frame.shape, self.xm_per_px, self.ym_per_px, self.keep_n_fits)
        if self.roi_mask is None:
            roi = lu.centered_trapezoid(frame, top_ratio = 0.2,height = 280, bottom_crop=35)
            self.roi_mask = lu.roi_mask(frame, roi)

        # Distortion correction
        undistorted = lu.undistort(frame, self.calib_params)
        # Binarization
        binarized = binarize(undistorted, self.roi_mask)
        # Perspective transformation
        warped = lu.warp_perspective(binarized, self.persp_params)
        # Lane detection
        new_lane = detect_lines(warped, self.lane)
        # Composing result frame
        result = self.lane.draw_lane(undistorted, self.persp_params)
        result = self.side_stack_imgs([result, np.dstack([binarized]*3),
                                       self.draw_finding_process(warped)])
        result = self.lane.draw_info(result,(970, 380))        
        self.lane = new_lane
        return result     
        
    def process_video(self, src_path, tgt_path):
        self.lane = None
        lu.process_video(src_path, self.process_frame, tgt_path)
        pass

calib_params_path='.\\camera_cal\\calib_params.p'
persp_params_path='.\\camera_cal\\persp_params.p'        
pipeline = Pipeline(calib_params_path, persp_params_path, 3.7/615, 3/118)
pipeline.process_video('project_video.mp4', 'project_result.mp4')
