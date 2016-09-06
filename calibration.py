import sys,os, glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lib')
import filter
import analyze_lib as lib

if __name__ == '__main__':
    filter.liner_calibration('carv/new/10_2_extracted_LPF.log', 'calib/variance_LPF.log', threshold=0.1)
