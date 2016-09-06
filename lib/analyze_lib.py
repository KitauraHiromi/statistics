import math, os
import numpy as np
import re

S = 5.5**2*math.pi

def vol_kPa_convert(vol, area):
    return (4.7246*vol + 0.1425) * 3.3/4096 /area * 1000

def vol_res_convert(v1, v2):
    if(v1>v2):
        tmp=v1-v2
    else:
        tmp=0

    if(v1>0xfc and tmp<5):
        tmp=0
    if(tmp==0):
        res=4095
    else:
        res=float(4095-v1)/tmp
    return res * 4700


def convertT(res):
    thr=[40, 30, 20, 10, 5]
    slope=[0.05, 0.1, 0.4, 0.85, 0.95]
    ret = 0
    if(res>thr[0]):
        ret = slope[0]*(4095.-res)/(4095-thr[0])
    elif(res>thr[1]):
        ret = (slope[1]-slope[0])*(thr[0]-res)/(thr[0]-thr[1])+slope[0]
    elif(res>thr[2]):
        ret = (slope[2]-slope[1])*(thr[1]-res)/(thr[1]-thr[2])+slope[1]
    elif(res>thr[3]):
        ret = (slope[3]-slope[2])*(thr[2]-res)/(thr[2]-thr[3])+slope[2]
    elif(res>thr[4]):
        ret = (slope[4]-slope[3])*(thr[3]-res)/(thr[3]-thr[4])+slope[3]
    else:
        ret - slope[4]
    return ret


def time_pressure_res(read_filename, area=S):
    read_file_splited = os.path.basename(read_filename).split('_')
    row, col = read_file_splited[0], read_file_splited[1].split('.')[0]
    __FIRST__ = True
    with open(read_filename, 'r') as read_file:
        line = read_file.readline()
        v1, v2, pres, force_volt , start_time= 0, 0, 0, 0, 0
    
        write_filename = read_filename.split('.')[0] + '_extracted.log' 
        with open(write_filename, 'w') as write_file:
            while line:
                tmp = line.split(', ')

                if(__FIRST__):
                    line = read_file.readline()
                    tmp = line.split(', ')
                    start_time = tmp[1]
                    __FIRST__ = False

                try:
                    if(tmp[2]=='fc' and tmp[3]=='1' and tmp[4]=='1' and tmp[5]=='1' and tmp[6]=='0'):
                        force_volt = float(tmp[9])
                    elif(tmp[2]=='fc' and tmp[3]=='1' and tmp[4]=='1' and tmp[5]== row and tmp[6]=='0'):
                        v1 = float(tmp[9])
                    elif(tmp[2]=='fc' and tmp[3]=='1' and tmp[4]=='1' and tmp[5]== row and tmp[6]== col):
                        v2 = float(tmp[9])
                        time = (float(tmp[1]) - float(start_time))*0.3*0.001
                        write_file.write(str(time)+' '+str(vol_kPa_convert(force_volt, area))+' '+str(vol_res_convert(v1, v2))+'\n')
                    
                    elif(tmp[2]=='fc' and tmp[3]=='1' and tmp[4]=='1' and tmp[5]== row and tmp[6]== str(int(col)-1)):
                        v2 = float(tmp[8])
                        time = (float(tmp[1]) - float(start_time))*0.3*0.001
                        write_file.write(str(time)+' '+str(vol_kPa_convert(force_volt, area))+' '+str(vol_res_convert(v1, v2))+'\n')
                    
                except IndexError:
                    pass
                    
                except ValueError:
                    print tmp[1]
                line = read_file.readline()


def time_pressure_res_tac(read_filename, area=S):
    read_file_splited = os.path.basename(read_filename).split('_')
    row, col = read_file_splited[0], read_file_splited[1].split('.')[0]
    __FIRST__ = True
    with open(read_filename, 'r') as read_file:
        line = read_file.readline()
        v1, v2, pres, force_volt , start_time= 0, 0, 0, 0, 0
        # old version
        #write_filename = read_filename.split('.')[0] + '_extracted.log' 

        # new version
        write_filename = read_filename.split('.')[0].replace('row', '') + 'extracted.log'
        with open(write_filename, 'w') as write_file:
            while line:
                tmp = line.split(', ')

                while __FIRST__:
                    try:
                        line = read_file.readline()
                        tmp = line.split(', ')
                        if tmp[1] != '\n':
                            start_time = tmp[1]
                            __FIRST__ = False
                    except IndexError:
                        line = read_file.readline()

                try:
                    if(tmp[2]=='fc' and tmp[3]=='1' and tmp[4]=='1' and tmp[5]=='1' and tmp[6]=='0'):
                        force_volt = float(tmp[9])
                    elif(tmp[2]=='fc' and tmp[3]=='1' and tmp[4]=='1' and tmp[5]== row and tmp[6]=='0'):
                        v1 = float(tmp[9])
                    elif(tmp[2]=='fc' and tmp[3]=='1' and tmp[4]=='1' and tmp[5]== row and tmp[6]== col):
                        v2 = float(tmp[9])
                        time = (float(tmp[1]) - float(start_time))*0.3*0.001
                        res = vol_res_convert(v1, v2)
                        write_file.write(str(time)+' '+str(vol_kPa_convert(force_volt, area))+' '+str(res) + ' ' + str(convertT(res/4700)) +'\n')
                    
                    elif(tmp[2]=='fc' and tmp[3]=='1' and tmp[4]=='1' and tmp[5]== row and tmp[6]== str(int(col)-1)):
                        v2 = float(tmp[8])
                        time = (float(tmp[1]) - float(start_time))*0.3*0.001
                        res = vol_res_convert(v1, v2)
                        write_file.write(str(time)+' '+str(vol_kPa_convert(force_volt, area))+' '+str(res) + ' ' + str(convertT(res/4700)) +'\n')
                    
                except IndexError:
                    pass

                except ValueError:
                    print start_time
                line = read_file.readline()


def std_ave_of_pres_res_tac(filename_list, width):
    tmp = []
    tmp2 = []
    for name in filename_list:
        with open(name, 'r') as read_file:
            for line in read_file:
                tmp.append([float(line.split(' ')[1]), float(line.split(' ')[2]), float(line.split(' ')[3])])
    tmp = np.array(sorted(tmp))
    #print tmp
    i = 0
    while (i+1)*width<len(tmp):
        b = i*width        #begin
        e = (i+1)*width    #end
        tmp2.append([np.average(tmp.T[0][b:e]), np.std(tmp.T[0][b:e]), np.average(tmp.T[1][b:e]), np.std(tmp.T[1][b:e]), np.average(tmp.T[2][b:e]), np.std(tmp.T[2][b:e])])
        i += 1
    return tmp2

def std_ave_of_time_pres_res_tac(filename_list, width):
    tmp = []
    tmp2 = []
    for name in filename_list:
        with open(name, 'r') as read_file:
            for line in read_file:
                s = line.split(' ')
                tmp.append([float(s[0]), float(s[1]), float(s[2]), float(s[3])])
    #sort by time
    tmp = np.array(sorted(tmp))
    i = 0
    while (i+1)*width<len(tmp):
        b = i*width        #begin
        e = (i+1)*width    #end
        tmp2.append([np.average(tmp.T[0][b:e]), np.average(tmp.T[1][b:e]), np.std(tmp.T[1][b:e]), np.average(tmp.T[2][b:e]), np.std(tmp.T[2][b:e]), np.average(tmp.T[3][b:e]), np.std(tmp.T[3][b:e])])
        i += 1
    return tmp2


def std_ave_of_pres_res(filename_list, width):
    tmp = []
    tmp2 = []
    for name in filename_list:
        with open(name, 'r') as read_file:
            for line in read_file:
                tmp.append([float(line.split(' ')[1]), float(line.split(' ')[2])])
    #sort by pressure
    tmp = np.array(sorted(tmp))
    i = 0
    while (i+1)*width<len(tmp):
        b = i*width
        e = (i+1)*width
        tmp2.append([np.average(tmp.T[0][b:e]), np.std(tmp.T[0][b:e]), np.average(tmp.T[1][b:e]), np.std(tmp.T[1][b:e])])
        i += 1
    return tmp2

def std_ave_of_pres(filename_list, width):
    tmp = []
    tmp2 = []
    for name in filename_list:
        with open(name, 'r') as read_file:
            for line in read_file:
                tmp.append(float(line.split(' ')[1]))
                
    tmp = np.array(tmp)
    i = 0
    while (i+1)*width<len(tmp):
        tmp2.append([np.average(tmp[i*width:(i+1)*width]), np.std(tmp[i*width:(i+1)*width])])
        i += 1
    return tmp2

def f_b_devide(filename_list):
    dev_time = 3.8
    for name in filename_list:

        #calculate dev_time
        with open(name, 'r') as read_file:
            time_stamps = []
            for line in read_file:
                tmp = line.split(' ')
                if float(tmp[1]) > 30:
                    time_stamps.append(float(tmp[0]))
            dev_time = np.average(np.array(time_stamps))

        #write file
        with open(name, 'r') as read_file:
            spl = name.split('.')[0]
            spl = re.sub('_extracted', '', spl)
            with open(spl + '_forward.log', 'w') as write_file1:
                with open(spl + '_back.log', 'w') as write_file2:
                    for line in read_file:
                        tmp = float(line.split(' ')[0])
                        try:
                            if tmp < dev_time:
                                write_file1.write(line)
                            else:
                                write_file2.write(line)
                        except IndexError:
                            pass
