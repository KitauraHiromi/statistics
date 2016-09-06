# Devide data into forward and back phase to evaluate hysterisys.
import sys,os, glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lib')
import analyze_lib as lib
import re

if __name__ == '__main__':
    filename_list = []
    dir = 'calib/*'
    for name in glob.glob(dir):
        if re.match(dir + '[\d_]+_cariv_extracted_LPF.log', name):
            filename_list.append(name)
    print filename_list

    lib.f_b_devide(filename_list)
