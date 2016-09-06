import sys,os, glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lib')
import analyze_lib as lib
import re

if __name__ == '__main__':
    filename_list = []
    dir = 'calib/'
    for name in glob.glob(dir+'*'):
        try:
            if re.match(dir+'[\d_]+_cariv_extracted_LPF.log', name):
            #if name.split('_')[-1] == 'LPF.log' and name.split('_')[-2] == 'extracted':
                if not re.match('5_1', name):
                    filename_list.append(name)
        except IndexError:
            pass
    print filename_list
    width = 3000
    #tmp1 = lib.std_ave_of_res(filename_list, width)
    #tmp2 = lib.std_ave_of_pres(filename_list, width)
    tmp = lib.std_ave_of_pres_res(filename_list, width)
    with open(dir+"variance_LPF.log", 'w') as write_file:
    #with open('20160523/variance_LPF.log', 'w') as write_file:
        #print tmp
        for i in range(len(tmp)):
            #pres-ave pres-std res-ave res-std 
            write_file.write(str(tmp[i][0]) + " " + str(tmp[i][1]) + " " + str(tmp[i][2]) + " " + str(tmp[i][3]) + "\n")
