#coding: utf-8
import analyze_lib as lib
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
# from lib.overload import *

# using line_profiler
# INSTALL
# pip install line_profile
# git clone https://github.com/rkern/line_profiler.git
# USAGE
# python line_profiler/kernprof.py -l -v hoge.py

def LPF(t, y, fp, fs):
    '''
    [Arguments]
    (R_list t, R_list y, R fp, R fs)

    [Return]
    (R_list t, R_list y)

    [Description]
    Low Pass Filter.
    fp is 通過域端周波数[Hz], fs is 阻止域端周波数[Hz].
    Data should be the set of (time t, value y).
    You may change the inner parameter according to your data.
    '''
    sampling_time = 0.015
    fn = 1/(2*sampling_time)
    # パラメータ設定
    #fp = 1                          # 通過域端周波数[Hz]
    #fs = 2                          # 阻止域端周波数[Hz]
    gpass = 1.0                       # 通過域最大損失量[dB]
    gstop = 40.0                      # 阻止域最小減衰量[dB]
    # 正規化
    Wp = fp/fn
    Ws = fs/fn

    # ローパスフィルタで波形整形
    # バターワースフィルタ
    N, Wn = signal.buttord(Wp, Ws, gpass, gstop)
    b1, a1 = signal.butter(N, Wn, "low")
    y1 = signal.filtfilt(b1, a1, y)

    '''
    # プロット
    plt.figure()
    plt.plot(t, y, "b")
    plt.plot(t, y1, "r", linewidth=2, label="butter")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()
    '''
    return t, y1


def Smoothing_Filter(data, kernel_size=3, __MODE__="gaussian"):
    if __MODE__ == "gaussian":
        return ndimage.gaussian_filter(data, sigma=kernel_size)
    elif __MODE__ == "median":
        return ndimage.median_filter(data, kernel_size)


def Sharpen_Filter(data, alpha=30):
        blurred = ndimage.gaussian_filter(data, 1)
        return data + alpha * (data - blurred)


def Binary_Division(data):
    val = filters.threshold_otsu(data)
    mask = data > val
    return mask * data


def edge_detection(data):
    if isinstance(data[0], list) or isinstance(data[0], np.ndarray):
        sx = ndimage.sobel(data, axis=0, mode="constant")
        sy = ndimage.sobel(data, axis=1, mode="constant")
        sob = np.hypot(sx, sy)
    else:
        sob = ndimage.sobel(data, axis=0, mode="constant")
    return sob


def window_function(data_list, begin, end):
    # data_list ... [ 'time pressure resistance', ..., 't p r' ] which is from line.readlines()
    # element ... 'time pressure resistance'
    # tmp_element ... [ time, pressure, resistance ] 
    tmp_list = []
    for element in data_list:
        tmp_element = map(float, element.split(' '))
        if begin < tmp_element[1] < end:
            tmp_list.append(tmp_element)
    return tmp_list


def window_function_with_index(data_list, index, begin, end):
    # data_list ... [ 'time pressure resistance', ..., 't p r' ] which is from line.readlines()
    # element ... 'time pressure resistance'
    # tmp_element ... [ time, pressure, resistance ] 
    tmp_list = []
    for element in data_list:
        tmp_element = map(float, element.split(' '))
        if begin < tmp_element[index] < end:
            tmp_list.append(tmp_element)
    return tmp_list


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


def ellipse_pivot(data, ellipse):
    center, size, theta = ellipse
    ySize, xSize = map(int, data.shape)
    for j in xrange(ySize):
        bottom, top = map(int, rotated_ellipse_bound_y(j, center, size, theta))

        if 0 < bottom < xSize:
            data[j][:bottom] = 0
        if 0 < top < xSize:
            data[j][top:] = 0
    return data


def circle_pivot(data, center, r):
    r = int(r)
    x, y = map(int, center)
    for i in xrange(y-r):
        data[i][:] = 0

    for i in xrange(y-r, y+r):
        h = int(math.sqrt(r**2 - (x-i)**2))
        data[i][:x - h] = 0
        data[i][x + h:] = 0

    for i in xrange(y+r, len(data)):
        data[i][:] = 0
    return data

'''
should be exported to another lib
'''

def between_peaks_in_line_filter(_data):
    xSize = ySize = len(_data)
    mean = _data.mean()
    # print mean
    th = 2 * 10**6
    if mean < th:
        return [0 for i in xrange(len(_data))]

    filtered_data = _data / (mean + 1)
    # if _data.mean() > 0.1:
        # filtered_data = Binary_Division(data)
    # else:
        # filtered_data = data
    filtered_data = Smoothing_Filter(filtered_data, kernel_size=15, __MODE__="gaussian")
    sob = edge_detection(filtered_data)
    tmp = [i for i in range(xSize) if sob[i] < 0.1]
    # time consuming
    tmp = [i for i in tmp if i-1 not in tmp and 200 < i < 1800]
    try:
        bottom, top = tmp[0], tmp[-1]
    except IndexError:
        bottom, top = 800, 800
    return [_data[i] if bottom < i < top else 0 for i in xrange(len(_data))]


def between_peaks_in_array_filter(_data):
    xSize = ySize = int(math.sqrt(len(_data)))
    print xSize, ySize
    mean = _data.mean()
    filtered_data = _data / (mean + 1)
    filtered_data = Smoothing_Filter(filtered_data, kernel_size=15, __MODE__="gaussian")
    sob = edge_detection(filtered_data)
    sob = edge_detection(sob)

    # assuming len(sob) == len(_data)
    for i in xrange(len(sob)):
        tmp = [j for j in range(xSize) if sob[i][j] < 0.1]
        # time consuming
        tmp = [j for j in tmp if j-1 not in tmp and 200 < j < 1800]
        try:
            bottom, top = tmp[0], tmp[-1]
        except IndexError:
            bottom, top = 800, 800
        _data[i] = [_data[i][j] if bottom < j < top else 0 for j in xrange(len(_data[i]))]
    return _data


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
    contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    for cnt in contours:
        # exclude data of having a few points
        if len(cnt) < 1000:
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
    except ValueError:
        max_ellipse = None
    return data, max_ellipse

def EM_Algorithm(data, n_components=1, covariance_type="diag"):
    gmm = GMM(n_components, covariance_type)
    gmm.fit(data)
    return gmm.weights_, gmm.means_, gmm.covars_


def draw_multivariate_gaussian(x, weights, means, covars, covariance_type="diag"):
    var = []
    for i in xrange(len(means)):
        var.append(weights[i] * multivariate_normal.pdf(x, mean=means[i], cov=covars[i]))
    return var



def liner_parm_calc(x1, y1, x2, y2):
    '''
    [Arguments]
    (R x1, R y1, R x2, R y2)
    Two cartesian coordinates points, p1(x1, y1), p2(x2, y2)

    [Return]
    (float slope, float intercept)

    [Description]
    Return slope and intercept of the line which pass through p1 and p2.

    [Warning]
    If input vertical line, the accuracy will be low.
    '''
    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
    try:
        a = (y2-y1)/(x2-x1)
    except ZeroDivisionError:
        a = 100000000000
    b = y1 - a*x1
    return a, b

def liner_conv(target_a, target_b, a, b, plot):
    tmp = [plot[0], target_a/a*(plot[1] - b) + target_b]
    return tmp

def line_func(x, a, b):
    '''
    [Arguments]
    (R x, R a, R b)

    [Return]
    (float y)

    [Description]
    Return y (y=ax+b)
    '''
    return float(x)*float(a)+float(b)

def exp1(x, a, b, c, d, e):
    '''
    [Arguments]
    (R x, Ra, R b, R c, R d, R e)

    [Return]
    (float y)

    [Description]
    Return y (y = a*exp{b*x^3} + c*exp{d*x^2} + e)
    '''
    return a*np.exp(b*x**3) + c*np.exp(d*x**2) + e

def exp2(x, a, b, c):
    '''
    [Arguments]
    (R x, Ra, R b, R c)

    [Return]
    (float y)

    [Description]
    Return y (y = a*exp{b*x^3})
    '''
    return a*np.exp(b*x**3) + c

def exp(x, a, b, c):
    '''
    [Arguments]
    (R x, Ra, R b, R c,)

    [Return]
    (float y)

    [Description]
    Return y (y = a*exp{b*x} + c)
    '''
    return a*np.exp(b*x) + c

def func_inverse(y, a, b, c, d):
    x_3 = math.log(math.fabs(y-d)/a)
    return math.pow(x_3 ,1.0/3)


def fitting(filename, func):
    '''
    [Arguments]
    (str filename, function func)

    [Return]
    (list parameter_of_function)

    [Description]
    Optimize the parameters of the given function according to the data in input file by scipy.optimize.curve_fit.
    '''

    #filename = variance_LPF.log
    # x < 0.7
    xdata = []
    ydata = []
    parameter_initial = np.array([0.0]*(func.func_code.co_argcount-1))
    with open(filename, 'r') as read_file:
        for line in read_file:
            tmp = line.split(' ')
            xdata.append(float(tmp[0]))
            ydata.append(float(tmp[2]))
        xdata = np.array(xdata)
        ydata = np.array(ydata)
    par_opt, covariance = scipy.optimize.curve_fit(func, xdata, ydata, p0=parameter_initial)
    #y = func(xdata, *par_opt)
    #plt.plot(xdata, ydata, 'o')
    #plt.plot(xdata, y, '-')
    #plt.show()
    return par_opt

def convert_nonliner_to_liner(filename_list, filter_file):
    threshold = 1
    par_exp = fitting(filter_file, threshold)
    #line_inter = interpolate_interface(filter_file, 'liner', threshold-0.3)
    #spline = spline_interpolate(filter_file, threshold)
    tmp_val = 0
    for name in filename_list:
        with open(name, 'r') as read_file:
            with open(name.split('.')[0].replace("_extracted", "") + '_linered2.log', 'w') as write_file:

                for line in read_file:
                    tmp = line.split(' ')
                    try:
                        x = float(tmp[1])
                        if x < threshold:
                            #tmp_val = float(tmp[2])*x/exp1(x, par_exp[0], par_exp[1], par_exp[2], par_exp[3], par_exp[4])
                            #tmp_val = float(tmp[2])*x/exp2(x, par_exp[0], par_exp[1], par_exp[2])
                            pass
                        else:
                            #tmp_val = float(tmp[2])*x/line_splev(x, line_inter)
                            tmp_val = np.log(float(tmp[2])-par_exp[2])
                            #print tmp_val, exp(x, par_exp[0], par_exp[1], 0)
                        #print tmp_val
                        write_file.write(tmp[0] + ' ' + tmp[1] + ' ' + tmp[2] + ' ' + str(tmp_val) + '\n')

                    except IndexError:
                        pass

def spline_interpolate(xdata, ydata, min_x, max_x, show=True):
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    spline = scipy.interpolate.splrep(xdata, ydata, s=0, k=3)
    xn = np.arange(min_x, max_x, 0.1)
    yn = scipy.interpolate.splev(xn, spline, der=0)

    # show plot
    if show is True:
        plt.plot(xdata, ydata, 'o')
        plt.plot(xn, yn, '-')
        plt.show()

    return spline, [xdata, ydata, xn, yn]


def line_splev(x, line_inter):
    i = 0
    for j in range(len(line_inter)):
        i = j
        if line_inter[i][0] > x:
            break
    return line_func(x, line_inter[i][1], line_inter[i][2])

def liner_interpolate(xdata, ydata, min_x, max_x, show=True):
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    line_inter = [] #[[division, par1, par2], ...]

    xn = np.arange(min_x, max_x, 0.1)
    yn = []
    for i in range(len(xdata)-1):
        tmp = liner_parm_calc(xdata[i], ydata[i], xdata[i+1], ydata[i+1])
        line_inter.append([xdata[i+1], tmp[0], tmp[1]])

    for i in range(len(xn)):
        yn.append(line_splev(xn[i], line_inter))

    # show plot
    if show is True:
        plt.plot(xdata, ydata, 'o')
        plt.plot(xn, yn, '-')
        plt.show()

    return line_inter, [xdata, ydata, xn, yn]

def interpolate_interface(filename, __MODE__ , division=None, spline=None, th=0.6, min_x=0.6, max_x=27, show=True, __MEAN__=False):
    xdata = []
    ydata = []
    yn = []

    # index setting
    pres_index = 1
    res_index = 2
    if re.match('variance', os.path.basename(filename)):
        pres_index = 0
        res_index = 2

    # making plot for interpolating
    if division is None:
        with open(filename, 'r') as read_file:
            pre_x = 0
            N = 2
            i = 0
            stock_x = []
            stock_y = []
            for line in read_file:
                tmp = line.split(' ')
                x = float(tmp[pres_index])
                y = float(tmp[res_index])
                if min_x < x < max_x and x > pre_x:
                    if __MEAN__ is True:
                        stock_x.append(x)
                        stock_y.append(y)
                        i += 1
                        if i > N:
                            stock_x = np.array(stock_x)
                            stock_y = np.array(stock_y)
                            xdata.append(np.average(stock_x))
                            ydata.append(np.average(stock_y))
                            i = 0
                            stock_x = []
                            stock_y = []

                    else:
                        xdata.append(x)
                        ydata.append(y)
                    pre_x = x
        print xdata
    else:
        for i in range(len(division)):
            division_x = float(division[i])
            if min_x < division_x < max_x:
                xdata.append(division_x)
                ydata.append(scipy.interpolate.splev(division_x, spline))
    if __MODE__ == 'liner':
        return liner_interpolate(xdata, ydata, min_x, max_x, show)
    elif __MODE__ == 'spline':
        return spline_interpolate(xdata, ydata, min_x, max_x, show)


def liner_calibration(filename, average_data_filename, show=True, threshold=0.6):
    # generate the target liner interpolate
    target_line_inter, inter_plot1 = interpolate_interface(average_data_filename, 'liner', th=threshold, show=False)
    #target_spline, spline_plot1 = interpolate_interface(average_data_filename, 'spline', th=threshold, show=False)

    # generate line interpolate to calibrate
    division = np.array(target_line_inter).T[0]
    division_spline, inter_plot2 = interpolate_interface(filename, 'spline',th=threshold, show=False, __MEAN__=True)
    line_inter, inter_plot3 = interpolate_interface(filename, 'liner', division, division_spline, th=threshold, show=False)

    # show plot
    if show is True:
        plt.plot(inter_plot1[0], inter_plot1[1], 'o')
        plt.plot(inter_plot1[2], inter_plot1[3], '-')
        #plt.plot(inter_plot2[0], inter_plot2[1], 'o')
        plt.plot(inter_plot2[2], inter_plot2[3], '-')
        plt.plot(inter_plot3[0], inter_plot3[1], 'o')
        plt.plot(inter_plot3[2], inter_plot3[3], '-')
        plt.show()        

    with open(filename, 'r') as read_file:
        with open(filename.split('.')[0] + '_calibrated.log', 'w') as write_file:
            time_index = 0
            pres_index = 1
            res_index = 2
            for line in read_file:
                tmp = line.split(' ')
                plot = [float(tmp[pres_index]), float(tmp[res_index])]
                if plot[0] > threshold:
                    # search the variables of the line in which plot exists.
                    i = 0
                    for j in range(len(line_inter)):
                        i = j
                        if line_inter[j][0] > plot[0]:
                            break
                    slope, intercept = line_inter[j][1], line_inter[j][2]
                    target_slope, target_intercept = target_line_inter[j+1][1], target_line_inter[j+1][2]
                    print line_inter[j][0], target_line_inter[j+1][0]
                    # convert plots based on disired variables.
                    #converted_plot = liner_conv(target_slope, target_intercept, slope, intercept, plot)
                    converted_plot = [plot[0], line_func(plot[0], target_slope, target_intercept)/line_func(plot[0], slope, intercept)*plot[1]]
                    #converted_plot = [plot[0], scipy.interpolate.splev(plot[0], target_spline)/scipy.interpolate.splev(plot[0], spline_inter)*plot[0]]
                    if converted_plot[1] > 0:
                        write_file.write(str(tmp[time_index]) + ' ' + str(converted_plot[0]) + ' ' + str(converted_plot[1]) + '\n')
                




'''
def nonliner_to_liner(filename_list):
    width = 1500
    value_list = lib.std_ave_of_pres_res(filename_list, width) #value_list = [ [x1,y1], [x2,y2],..,[xn.yn] ]
    target_a, target_b = -10000000, 10000000
    a_list = []
    b_list = []
    for i in range(len(value_list)-1):
        a,b = liner_func(value_list[i][0], value_list[i][2], value_list[i+1][0], value_list[i+1][2])
        a_list.append(float(a))
        b_list.append(float(b))
    for i in range (len(a_list)):
        print a_list[i], b_list[i], value_list[i][0], value_list[2]
    
    for name in filename_list:
        with open(name, 'r') as read_file:
            with open(name.split('.')[0] + '_linered.log', 'w') as write_file:
                for line in read_file:
                    tmp = line.split(' ')
                    i = 0
                    for j in range(len(value_list)):
                        if (float(tmp[1]) > value_list[j][0]):
                            i += 1
                            #print tmp[1], '>', value_list[j][0]
                    try:
                        tmp_val = target_a/a_list[i]*(float(tmp[2]) - b_list[i]) + target_b
                        #print tmp_val
                        write_file.write(tmp[0] + ' ' + tmp[1] + ' ' + tmp[2] + ' ' + str(tmp_val) + '\n')
                    except IndexError:
                        pass

def variance_handmade_filter(plot):
    plot = [float(plot[0]), float(plot[1])]
    a1 = -6435974
    b1 = 20920060
    a2 = -19235
    b2 = 654779
    x1 = 0.78
    y1 = 15900000
    x2 = 3.15819
    y2 = 594030
    x3 = 33.75
    y3 = 5881

    target_a = (y1-y3)/(x1-x3)
    target_b = y1 - target_a*x1

    if plot[0] < x1:
        return plot
    elif plot[0] < x2:
        return liner_conv(target_a, target_b, a1, b1, plot)
    elif plot[0] < x3:
        return liner_conv(target_a, target_b, a2, b2, plot)
    else:
        return plot

def nonliner_to_liner_handmade(filename_list):
    for name in filename_list:
        with open(name, 'r') as read_file:
            with open(name.split('.')[0] + '_linered.log', 'w') as write_file:
                for line in read_file:
                    tmp = line.split(' ')
                    plot = [tmp[1], tmp[2]]
                    try:
                        plot = variance_handmade_filter(plot)
                        #print tmp_val
                        write_file.write(tmp[0] + ' ' + tmp[1] + ' ' + tmp[2] + ' ' + str(plot[1]) + '\n')
                    except IndexError:
                        pass
'''
