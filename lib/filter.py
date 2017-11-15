#coding: utf-8
import numpy as np
import scipy.optimize
import matplotlib.pylab as plt
import math
from scipy import signal
from scipy import ndimage
from skimage import filters
from sklearn.mixture import GMM
from scipy.stats import multivariate_normal
import scipy.interpolate
import os
import re
import cv2
import sys
from copy import copy as shallow_copy

def is_larger_than_fx(f, th, *args):
    return f(*args) < th


def in_ellipse(el, x, y):
    # el:[center_xy, size_ab, angle]
    return is_larger_than_fx(rotated_ellipse, 1, (x, y), el[0], el[1], el[2])


def ellipse(x, y, cx, cy, a, b, *args):
    x, y, cx, cy, a, b = map(float, [x, y, cx, cy, a, b])
    # if a, b is not long, sort axis radius
    a /= 2.
    b /= 2.
    return ((x - cx)/a)**2 + ((y - cy)/b)**2


def rotate_deg(pos, center, angle):
    x, y = pos
    cx, cy = center
    return ((x-cx)*math.cos(angle/180.*math.pi)+cx, (y-cy)*math.sin(angle/180.*math.pi)+cy)


def rotated_ellipse(pos, center, ab, angle, *args):
    x, y = rotate_deg(pos, center, -1*angle)
    cx, cy  = center
    a, b = ab
    return ellipse(x, y, cx, cy, a, b)


def circle_bound(y, center, r):
    y, r = map(float, (y, r))
    cx, cy = map(float, center)
    D = r**2 - (y-cy)**2
    if D > 0:
        h = math.sqrt(D)
        return (cx-h, cx+h)
    return (2, 1)

def ellipse_bound(y, center, size):
    a, b = map(float, size)
    a /= 2.
    b /= 2.
    y = float(y)
    cx, cy = map(float, center)
    D = 1 - ((y-cy)/b)**2
    if D > 0:
        h = a * math.sqrt(D)
        return (cx-h, cx+h)
    return (2, 1)


def rotated_ellipse_bound(x, center, size, theta):
    a, b = map(float, size)
    a /= 2.
    b /= 2.
    x = float(x)
    cx, cy = map(float, center)
    # Ay^2 + Bxy + Cy^2 = 1
    A = (math.sin(theta)/a)**2 + (math.cos(theta)/b)**2
    B = math.sin(2*theta) * (1/b**2 - 1/a**2)
    C = (math.cos(theta)/a)**2 + (math.sin(theta)/b)**2

    D = (B**2 - 4*A*C) * (x-cx)**2 + 4*A
    if D > 0:
        h = math.sqrt(D) / (2*A)
        t = -B/(2*a)*x
        return (cy + t-h, cy + t+h)
    return (2, 1)

def rotated_ellipse_bound_y(y, center, size, deg):
    a, b = map(float, size)
    a /= 2.
    b /= 2.
    cx, cy = map(float, center)
    y = float(y)
    rad = -deg / 180. * math.pi
    # Ax^2 + Bxy + Cx^2 = 1
    A = (math.cos(rad)/a)**2 + (math.sin(rad)/b)**2
    B = math.sin(2*rad) * (1/b**2 - 1/a**2)
    C = (math.sin(rad)/a)**2 + (math.cos(rad)/b)**2

    D = (B**2 - 4*A*C) * (y-cy)**2 + 4*A
    if D > 0:
        h = math.sqrt(D) / (2*A)
        t = -B/(2*A) * (y-cy)
        return (cx + t-h, cx + t+h)
    return (2, 1)


def circle_pivot(data, center, r, val_in=1, val_out=0):
    r = int(r)
    x, y = map(int, center)
    for i in range(y-r):
        data[i][:] *= val_out

    for i in range(y-r, y+r):
        h = int(math.sqrt(r**2 - (x-i)**2))
        data[i][:x - h] *= val_out
        data[i][x - h: x + h] *= val_in
        data[i][x + h:] *= val_out

    for i in range(y+r, len(data)):
        data[i][:] *= val_out
    return data
    

def ellipse_pivot(data, ellipse):
    center, size, theta = ellipse
    ySize, xSize = map(int, data.shape)
    for j in range(ySize):
        bottom, top = map(int, rotated_ellipse_bound_y(j, center, size, theta))

        if 0 < bottom < xSize:
            data[j][:bottom] = 0
        if 0 < top < xSize:
            data[j][top:] = 0
    return data

def ellipse_fitting(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if data.dtype is not np.uint8:
        # time consuming
        # data /= (max(data.flatten()) / 255)
        data = data.astype(np.uint8)
        # test show
        # plt.imshow(data)
        # plt.show()

    # contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # contours, hierarchy = [], []
    # test(contours, hierarchy)
    print(len(cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)))
    _, contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    for cnt in contours:
        # exclude data of having few points
        if len(cnt) < 250:
            continue

        # excluing ellipse whose size is small
        ellipse = cv2.fitEllipse(cnt)
        # ellipse: ((center_x, center_y), (a, b), angle)
        ellipses.append(ellipse)
        # if min(ellipse[1]) < 200:
            # continue
        # cv2.ellipse(data, ellipse, (255, 255, 255), 2)


    # ellipse = cv2.fitEllipse(contours[1])
    # im = cv2.ellipse(data, ellipse, (0, 255, 0), 2)
    # plt.imshow(data)
    # plt.show()
    dists = map(np.linalg.norm, [row[1] for row in ellipses])
    try:
        max_ellipse = ellipses[np.argmax(dists)]
    except IndexError:
        max_ellipse = None
    return data, max_ellipse