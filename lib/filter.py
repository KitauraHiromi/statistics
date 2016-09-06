#coding: utf-8
import analyze_lib as lib
import numpy as np
import scipy.optimize
import matplotlib.pylab as plt
import math
from scipy import signal
import scipy.interpolate
import os
import re

def liner_parm_calc(x1, y1, x2, y2):
    '''
    y = ax+b
    '''
    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)
    try:
        a = (y2-y1)/(x2-x1)
    except ZeroDivisionError:
        a = 100000000000
    b = y1 - a*x1
    #print a, b
    return a, b

def liner_conv(target_a, target_b, a, b, plot):
    tmp = [plot[0], target_a/a*(plot[1] - b) + target_b]
    #print plot[1], tmp[1]
    return tmp

def line_func(x, a, b):
    return float(x)*float(a)+float(b)

def exp1(x, a, b, c, d, e):
    return a*np.exp(b*x**3) + c*np.exp(d*x**2) + e

def exp2(x, a, b, c):
    return a*np.exp(b*x**3) + c

def exp(x, a, b, c):
    return a*np.exp(b*x) + c

def func_inverse(y, a, b, c, d):
    x_3 = math.log(math.fabs(y-d)/a)
    return math.pow(x_3 ,1.0/3)

def LPF(t, y, fp, fs):
    sampling_time = 0.015
    fn = 1/(2*sampling_time)
    # パラメータ設定
    #fp = 1                          # 通過域端周波数[Hz]
    #fs = 2                          # 阻止域端周波数[Hz]
    gpass = 1                       # 通過域最大損失量[dB]
    gstop = 40                      # 阻止域最小減衰量[dB]
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
    
    

def fitting(filename, th=0.7):
    #filename = variance_LPF.log
    # x < 0.7
    xdata = []
    ydata = []
    #parameter_initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    parameter_initial = np.array([0.0, 0.0, 0.0, 0.0])
    with open(filename, 'r') as read_file:
        for line in read_file:
            tmp = line.split(' ')
            if float(tmp[0]) >th:
                xdata.append(float(tmp[0]))
                ydata.append(float(tmp[2]))
        xdata = np.array(xdata)
        ydata = np.array(ydata)
    par_opt, covariance = scipy.optimize.curve_fit(exp, xdata, ydata, p0=parameter_initial)
    #print par_opt
    #y = exp1(xdata,par_opt[0],par_opt[1], par_opt[2], par_opt[3], par_opt[4])
    y = exp(xdata, par_opt[0],par_opt[1], par_opt[2])
    plt.plot(xdata, ydata, 'o')
    plt.plot(xdata, y, '-')
    plt.show()
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
