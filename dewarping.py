import cv2
from scipy import ndimage, interpolate
import numpy as np
import scipy.integrate as integrate
from tqdm import tqdm
from itertools import product

def poly(points):
        xs = list(map(lambda x: x[0], points))
        ys = list(map(lambda x: x[1], points))
        poly_full = np.poly1d(np.polyfit(xs, ys, min(4, len(points) - 1)))
        
        return poly_full
    
def dewarp_by_lines(path, top_line, bottom_line, points):
    top_left, top_right, bottom_left, bottom_right = points
        
    top_poly = poly(top_line)
    bottom_poly = poly(bottom_line)
    
    top_der = np.polyder(top_poly, 1)
    bottom_der = np.polyder(bottom_poly, 1) #getting edge polynoms
    
    top_l = lambda y: integrate.quad(lambda x: np.sqrt(1 + top_der(x) ** 2),  #length calculating functions
                                     top_left[0], y)[0]
    bottom_l = lambda y: integrate.quad(lambda x: np.sqrt(1 + bottom_der(x) ** 2),
                                        bottom_left[0], y)[0] 

    top_len = top_l(top_right[0])
    bottom_len = bottom_l(bottom_right[0]) 

    left_len = np.sqrt(np.sum(top_left - bottom_left) ** 2)
    right_len = np.sqrt(np.sum(top_right - bottom_right) ** 2)

    width = min(top_len, bottom_len)
    height = min(left_len, right_len)
    
    ratio_x = lambda z : (top_l(z) / top_len)
    ratio_y = lambda z : (bottom_l(z) / bottom_len)

    bot_cur = int(bottom_left[0])

    res = []

    for top in range(int(top_left[0]), int(top_right[0]) + 1): #here biection between top and bottom points is calculated 
        ratio = ratio_x(top)

        while(ratio_y(bot_cur) < ratio):
            bot_cur += 1

        res.append((ratio, top, bot_cur))
        bot_cur += 1
        
        
    w = int(top_left[0] + width + 1) - int(top_left[0])
    h = int(top_left[1] + height + 1) - int(top_left[1])
    

    pixels = cv2.imread(path, cv2.IMREAD_COLOR)
    
    final_pixels = np.zeros((h, w, 3), np.uint8)
    res = np.array(res)
    
    def inv_im(pair):
        x, y = pair
        ratio, top, bot = res[np.nonzero(abs(x) / width - res[:, 0]  < 1e-3)][0]
        part = y / height
        
        lambda_ = part / (1 - part)
        new_x = min(int((top_poly(top) + lambda_ * bottom_poly(bot)) // (1 + lambda_)), pixels.shape[0] - 1)
        new_x = max(new_x, 0)
        new_y = min(int((top + lambda_ * bot) // (1 + lambda_)), pixels.shape[1] - 1)
        new_y = max(new_y, 0)
        return pixels[new_x, new_y]

    indices = np.array(np.meshgrid(np.arange(w), np.arange(height))).T.reshape(-1, 2)
    final_pixels = np.apply_along_axis(inv_im, 1, indices)        
    
    
    return cv2.rotate(final_pixels.reshape(w, int(height), 3)[:, ::-1][:, :, ::-1], cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
