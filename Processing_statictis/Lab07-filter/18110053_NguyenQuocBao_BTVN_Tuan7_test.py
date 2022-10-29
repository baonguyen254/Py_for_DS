import numpy as np
import pandas as pd
import argparse
import math
import cv2
import sys 
from PIL import Image


def filter_bilateral( img_in, sigma_s, sigma_v, reg_constant=1e-8 ):
    """Simple bilateral filtering of an input image

    Performs standard bilateral filtering of an input image. If padding is desired,
    img_in should be padded prior to calling

    Args:
        img_in       (ndarray) monochrome input image
        sigma_s      (float)   spatial gaussian std. dev.
        sigma_v      (float)   value gaussian std. dev.
        reg_constant (float)   optional regularization constant for pathalogical cases

    Returns:
        result       (ndarray) output bilateral-filtered image

    Raises: 
        ValueError whenever img_in is not a 2D float32 valued numpy.ndarray
    """

    # check the input
    if not isinstance( img_in, np.ndarray ) or img_in.dtype != 'float32' or img_in.ndim != 2:
        raise ValueError('Expected a 2D np.ndarray with float32 elements')

    # make a simple Gaussian function taking the squared radius
    gaussian = lambda r2, sigma: (np.exp( -0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0

    # define the window width to be the 3 time the spatial std. dev. to 
    # be sure that most of the spatial kernel is actually captured
    win_width = int( 3*sigma_s+1 )

    # initialize the results and sum of weights to very small values for
    # numerical stability. not strictly necessary but helpful to avoid
    # wild values with pathological choices of parameters
    wgt_sum = np.ones( img_in.shape )*reg_constant
    result  = img_in*reg_constant

    # accumulate the result by circularly shifting the image across the
    # window in the horizontal and vertical directions. within the inner
    # loop, calculate the two weights and accumulate the weight sum and 
    # the unnormalized result image
    for shft_x in range(-win_width,win_width+1):
        for shft_y in range(-win_width,win_width+1):
            # compute the spatial weight
            w = gaussian( shft_x**2+shft_y**2, sigma_s )

            # shift by the offsets
            off = np.roll(img_in, [shft_y, shft_x], axis=[0,1] )

            # compute the value weight
            tw = w*gaussian( (off-img_in)**2, sigma_v )

            # accumulate the results
            result += off*tw
            wgt_sum += tw

    # normalize the result and return
    return result/wgt_sum



def bilateralfilter(image, texture, sigma_s, sigma_r):
    r = int(np.ceil(3 * sigma_s))
    # Image padding
    if image.ndim == 3:
        h, w, ch = image.shape
        I = np.pad(image, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.float32)
    elif image.ndim == 2:
        h, w = image.shape
        I = np.pad(image, ((r, r), (r, r)), 'symmetric').astype(np.float32)
    else:
        print('Input image is not valid!')
        return image
    # Check texture size and do padding
    if texture.ndim == 3:
        ht, wt, cht = texture.shape
        if ht != h or wt != w:
            print('The guidance image is not aligned with input image!')
            return image
        T = np.pad(texture, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.int32)
    elif texture.ndim == 2:
        ht, wt = texture.shape
        if ht != h or wt != w:
            print('The guidance image is not aligned with input image!')
            return image
        T = np.pad(texture, ((r, r), (r, r)), 'symmetric').astype(np.int32)
    # Pre-compute
    output = np.zeros_like(image)
    scaleFactor_s = 1 / (2 * sigma_s * sigma_s)
    scaleFactor_r = 1 / (2 * sigma_r * sigma_r)
    # A lookup table for range kernel
    LUT = np.exp(-np.arange(256) * np.arange(256) * scaleFactor_r)
    # Generate a spatial Gaussian function
    x, y = np.meshgrid(np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r)
    kernel_s = np.exp(-(x * x + y * y) * scaleFactor_s)
    # Main body
    if I.ndim == 2 and T.ndim == 2:     # I1T1 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[np.abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
                output[y - r, x - r] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1]) / np.sum(wgt)
    elif I.ndim == 3 and T.ndim == 2:     # I3T1 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
                wacc = np.sum(wgt)
                output[y - r, x - r, 0] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
                output[y - r, x - r, 1] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
                output[y - r, x - r, 2] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
    elif I.ndim == 3 and T.ndim == 3:     # I3T3 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 0] - T[y, x, 0])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 1] - T[y, x, 1])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 2] - T[y, x, 2])] * \
                      kernel_s
                wacc = np.sum(wgt)
                output[y - r, x - r, 0] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
                output[y - r, x - r, 1] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
                output[y - r, x - r, 2] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc
    elif I.ndim == 2 and T.ndim == 3:     # I1T3 filter
        for y in range(r, r + h):
            for x in range(r, r + w):
                wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 0] - T[y, x, 0])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 1] - T[y, x, 1])] * \
                      LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1, 2] - T[y, x, 2])] * \
                      kernel_s
                output[y - r, x - r] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1]) / np.sum(wgt)
    else:
        print('Something wrong!')
        return image

    # return np.clip(output, 0, 255)
    return output

#********************************************************

def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)

def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))

def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = diameter//2
    i_filtered = 0
    Wp = 0
    for i in range(diameter):
        for j in range(diameter):
            neighbour_x = int(x - (hl - i))
            neighbour_y = int(y - (hl - j))
            # if neighbour_x >= len(source):
            #     neighbour_x -= len(source)
            # if neighbour_y >= len(source[0]):
            #     neighbour_y -= len(source[0])
            gi = gaussian(source[x][y] - source[neighbour_x][neighbour_y], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = int(i_filtered)


def bilateral_filter_own(source, filter_diameter, sigma_i, sigma_s):
    img = cv2.cvtColor(source,cv2.COLOR_RGB2HSV)
    source = img[:,:,2]
    h, w = source.shape
    hl = filter_diameter
    new_h = h + hl * 2
    new_w = w + hl * 2
    new_image = np.zeros((new_h,new_w))
    new_image[hl: new_h - hl, hl: new_w - hl] = source
    filtered_image = np.zeros_like(new_image)
    for i in np.arange(hl, new_h - hl):
        for j in np.arange(hl, new_w - hl):
            apply_bilateral_filter(new_image, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
    img[:,:,2] = filtered_image[hl: new_h-hl,hl: new_w-hl]
    output = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return output

#*****************************************************

#**************************************************************************************************
#                                           Main
#**************************************************************************************************
def ex1():
    image = cv2.imread("t_rex.jpg")
    blurred = np.hstack([
    cv2.bilateralFilter(image, 5, 21, 21),
    cv2.bilateralFilter(image, 7, 31, 31),
    cv2.bilateralFilter(image, 9, 41, 41),
    cv2.bilateralFilter(image, 11, 51, 51)])
    cv2.imshow("Bilateral using cv2", blurred)
    # cv2.waitKey(0)


#*******

def ex2():
    image = cv2.imread("t_rex.jpg")
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # noise = np.random.random(img_gray.shape)
    # img_noise = img_gray.copy()
    # img_noise[noise < 0.01] = 0
    # img_noise[noise > 0.99] = 255
    # bilateral1 = bilateralfilter(image, img_gray, 1,0.1*255)
    bilateral1 = bilateralfilter(image, img_hsv, 51,51)
    cv2.imshow('blurred1', bilateral1)
    cv2.waitKey(0)

#*******
def ex3():
    image = cv2.imread("t_rex.jpg")
    bilateral2 = bilateral_filter_own(image, 9, 41,41)
    cv2.imshow('blurred2', bilateral2)
    cv2.imshow("Original", image)
    cv2.waitKey(0)
# def ex4():
#     image = cv2.imread("t_rex.jpg")
#     bilateral2 = apply_bilateral_filter(image, 5, 21,21)
#     cv2.imshow('blurred2', bilateral2)
#     cv2.waitKey(0)
# def ex5():
#     image = cv2.imread("t_rex.jpg")
#     bilateral2 = bilateral_filter(image, 7, 31,31)
#     cv2.imshow('blurred2', bilateral2)
#     cv2.waitKey(0)
def ex6():
    I = cv2.imread('t_rex.jpg').astype(np.float32)/255.0

    # bilateral filter the image
    B = np.stack([ 
        filter_bilateral( I[:,:,0], 41.0, 0.2 ),
        filter_bilateral( I[:,:,1], 41.0, 0.2 ),
        filter_bilateral( I[:,:,2], 41.0, 0.2 )], axis=2 )
    c = np.hstack( [I,B] )
    cv2.imshow('blurred1', c)
    cv2.waitKey(0)

if __name__ == "__main__":
    ex1()
    ex2()
    # ex3()
    # ex6()