import sys,os, glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lib')
import filter
import re
#make *_liner.log

if __name__ == '__main__':
    filename_list = []
    dir = 'test/*'
    for filename in glob.glob(dir):
        if re.match(dir+'[\d_]+extracted_LPF.log', filename):
            filename_list.append(filename)
    print filename_list
    filter.convert_nonliner_to_liner(filename_list, 'variance/variance_LPF.log')
