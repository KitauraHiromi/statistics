import sys,os, glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lib')
import filter
import analyze_lib as lib
import re
import math

def area_func(R):
    r = 5.5
    b = 0.5
    d = math.sqrt(4*R*b)
    th =math.atan(math.sqrt(4*r**2-d**2)/d)
    return (math.pi-2*th)*r**2 + d*math.sqrt(r**2-d**2/4)

def extract_log(dir):
    #making _extracted.log
    R = None
    for filename in glob.glob(dir):
        matching_rule1 = '[\d_abcalibg_N]*_row.log'
        matching_rule2 = '10_2_row.log'
        if re.match(dir + matching_rule1, filename):
            '''
            if re.search('[\d]+_[\d]+_[\d]+[abcd]',filename):
                R = int(re.findall('[\d]+', filename)[2])
            '''
            print filename
            lib.time_pressure_res_tac(filename)

def Monotonic_increase(float_list):
    n = 0
    pre_value = 0
    num_index_list = []
    #print float_list
    for i in range(len(float_list)):
        if float_list[i] - pre_value > 0:
            n += 1
        elif n > 0:
            num_index_list.append([n, i-n])
            n = 0
        else:
            n = 0
        pre_value = float_list[i]
    return num_index_list
 
def Monotonic_decrease(float_list):
    n = 0
    pre_value = 0
    num_index_list = []
    #print float_list
    for i in range(len(float_list)):
        if float_list[i] - pre_value < 0:
            n += 1
        elif n > 0:
            num_index_list.append([n, i])
            n = 0
        else:
            n = 0
        pre_value = float_list[i]
    return num_index_list

def LPF_pressure_test(filename):
    time = []
    press = []
    with open(filename, 'r') as read_file:
        for line in read_file:
            tmp = line.split(' ')
            time.append(float(tmp[0]))
            press.append(float(tmp[1]))

    filter.LPF(time, press)

def LPF_pressure_res(filename):
    # filename: *_extracted.log
    time = []
    press = []
    res = []
    with open(filename, 'r') as read_file:
        for line in read_file:
            tmp = line.split(' ')
            time.append(float(tmp[0]))
            press.append(float(tmp[1]))
            res.append(float(tmp[2]))

    t, p = filter.LPF(time, press, fp=5, fs=10)
    t, r = filter.LPF(time, res, fp=5, fs=10)
    r = res
    '''
    inc_list = Monotonic_increase(p)
    dec_list = Monotonic_decrease(p)

    start_index = max(inc_list)[1]
    end_index = max(dec_list)[1]

    for i in range(len(p)):
        if i < start_index:
            p[i] = t[i]/10
            #r[i] = 19246500
        elif i > end_index:
            p[i] = 3/t[i]
            #r[i] = 19246500
    '''
    with open(filename, 'r') as read_file:
        with open(filename.split('.')[0].replace('extracted' ,'')+ 'LPF.log', 'w') as write_file:
            i = 0
            for line in read_file:
                tmp = line.split(' ')
                write_file.write(str(t[i]) + ' ' + str(p[i]) + ' ' + str(r[i]) + '\n')
                i += 1



if __name__ == '__main__':
    #LPF_pressure('20160523/7_4_1_extracted.log')
    dir1 = 'carv/new/*'
    dir2 = 'test/*'
    dir3 = 'carv/20160624/*'
    dir4 = 'static_force/10_3/*'
    dir5 = 'hys/*'
    dir = dir5
    #extract_log(dir)

    for filename in glob.glob(dir):
        if filename.split('_')[-1] == 'unbiased.log':
            #print filename
            LPF_pressure_res(filename)
            #lib.f_b_devide([re.sub('extracted', 'LPF', filename)])
    
        
    #filename = input()
    #lib.time_pressure_res("datalog/11_2_test.log")
