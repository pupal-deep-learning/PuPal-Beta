import cv2
import numpy as np

def circler(im):
    ## find contours
    ret, thres = cv2.threshold(im, 127,255,0)
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 2:
        # iris
        cnt_iris = contours[0]
        (xi,yi), radius_iris = cv2.minEnclosingCircle(cnt_iris)
        center_i = (int(xi), int(yi))
        radius_i = int(radius_iris)

        # pupil
        cnt_pupil = contours[1]
        (x_p,y_p), radius_pupil = cv2.minEnclosingCircle(cnt_pupil)
        center_p = (int(x_p), int(y_p))
        radius_p = int(radius_pupil)

        # ratio
        ratio = round((radius_pupil / radius_iris), 4)
        if ratio > 0.2 and ratio < 0.8:
            return ratio, radius_i, center_i, radius_p, center_p
        else:
            pass

    else:
        pass


# convert non zeros to red
def colorise(img,rgbcode):
    lower =(1, 1, 1) # lower bound for each channel
    upper = (255, 255, 255) # upper bound for each channel

    # create the mask and use it to change the colors
    mask = cv2.inRange(img, lower, upper)
    img[mask != 0] = rgbcode

    return img


## create alpha layer
def alpha(img):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 #creating a dummy alpha channel image.
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    return img_BGRA
