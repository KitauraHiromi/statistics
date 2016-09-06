import sys,os, glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lib')
import statistics
import filter

if __name__ == '__main__':
    '''
    # a
    filenames1 = [ 'carv/20160624/10_2_10a1_LPF.log', 'carv/20160624/10_2_10a2_LPF.log', 'carv/20160624/10_2_10a3_LPF.log', 'carv/20160624/10_2_10a4_LPF.log', 'carv/20160624/10_2_10a5_LPF.log', 'carv/20160624/10_2_10a6_LPF.log', 'carv/20160624/10_2_10a7_LPF.log', 'carv/20160624/10_2_10a8_LPF.log', 'carv/20160624/10_2_10a9_LPF.log', 'carv/20160624/10_2_10a10_LPF.log' ]
    filenames2 = [ 'carv/20160624/10_2_20a1_LPF.log', 'carv/20160624/10_2_20a2_LPF.log', 'carv/20160624/10_2_20a3_LPF.log', 'carv/20160624/10_2_20a4_LPF.log', 'carv/20160624/10_2_20a5_LPF.log', 'carv/20160624/10_2_20a6_LPF.log', 'carv/20160624/10_2_20a7_LPF.log', 'carv/20160624/10_2_20a8_LPF.log', 'carv/20160624/10_2_20a9_LPF.log', 'carv/20160624/10_2_20a10_LPF.log' ]
    filenames3 = [ 'carv/20160624/10_2_40a1_LPF.log', 'carv/20160624/10_2_40a2_LPF.log', 'carv/20160624/10_2_40a3_LPF.log', 'carv/20160624/10_2_40a4_LPF.log', 'carv/20160624/10_2_40a5_LPF.log', 'carv/20160624/10_2_40a6_LPF.log', 'carv/20160624/10_2_40a7_LPF.log', 'carv/20160624/10_2_40a8_LPF.log', 'carv/20160624/10_2_40a9_LPF.log', 'carv/20160624/10_2_40a10_LPF.log' ]
    filenames4 = [ 'carv/20160624/10_2_80a1_LPF.log', 'carv/20160624/10_2_80a2_LPF.log', 'carv/20160624/10_2_80a3_LPF.log', 'carv/20160624/10_2_80a4_LPF.log', 'carv/20160624/10_2_80a5_LPF.log', 'carv/20160624/10_2_80a6_LPF.log', 'carv/20160624/10_2_80a7_LPF.log', 'carv/20160624/10_2_80a8_LPF.log', 'carv/20160624/10_2_80a9_LPF.log', 'carv/20160624/10_2_80a10_LPF.log' ]
    filenames5 = [ 'carv/20160624/10_2_1_LPF.log', 'carv/20160624/10_2_2_LPF.log', 'carv/20160624/10_2_3_LPF.log', 'carv/20160624/10_2_4_LPF.log', 'carv/20160624/10_2_5_LPF.log', 'carv/20160624/10_2_6_LPF.log', 'carv/20160624/10_2_7_LPF.log', 'carv/20160624/10_2_8_LPF.log', 'carv/20160624/10_2_9_LPF.log', 'carv/20160624/10_2_10_LPF.log' ]
    '''

    '''
    # b
    filenames6 = [ 'carv/20160624/10_2_10b1_LPF.log', 'carv/20160624/10_2_10b2_LPF.log', 'carv/20160624/10_2_10b3_LPF.log', 'carv/20160624/10_2_10b4_LPF.log', 'carv/20160624/10_2_10b5_LPF.log', 'carv/20160624/10_2_10b6_LPF.log', 'carv/20160624/10_2_10b7_LPF.log', 'carv/20160624/10_2_10b8_LPF.log', 'carv/20160624/10_2_10b9_LPF.log', 'carv/20160624/10_2_10b10_LPF.log' ]
    filenames7 = [ 'carv/20160624/10_2_20b1_LPF.log', 'carv/20160624/10_2_20b2_LPF.log', 'carv/20160624/10_2_20b3_LPF.log', 'carv/20160624/10_2_20b4_LPF.log', 'carv/20160624/10_2_20b5_LPF.log', 'carv/20160624/10_2_20b6_LPF.log', 'carv/20160624/10_2_20b7_LPF.log', 'carv/20160624/10_2_20b8_LPF.log', 'carv/20160624/10_2_20b9_LPF.log', 'carv/20160624/10_2_20b10_LPF.log' ]
    filenames8 = [ 'carv/20160624/10_2_40b1_LPF.log', 'carv/20160624/10_2_40b2_LPF.log', 'carv/20160624/10_2_40b3_LPF.log', 'carv/20160624/10_2_40b4_LPF.log', 'carv/20160624/10_2_40b5_LPF.log', 'carv/20160624/10_2_40b6_LPF.log', 'carv/20160624/10_2_40b7_LPF.log', 'carv/20160624/10_2_40b8_LPF.log', 'carv/20160624/10_2_40b9_LPF.log', 'carv/20160624/10_2_40b10_LPF.log' ]
    filenames9 = [ 'carv/20160624/10_2_80b1_LPF.log', 'carv/20160624/10_2_80b2_LPF.log', 'carv/20160624/10_2_80b3_LPF.log', 'carv/20160624/10_2_80b4_LPF.log', 'carv/20160624/10_2_80b5_LPF.log', 'carv/20160624/10_2_80b6_LPF.log', 'carv/20160624/10_2_80b7_LPF.log', 'carv/20160624/10_2_80b8_LPF.log', 'carv/20160624/10_2_80b9_LPF.log', 'carv/20160624/10_2_80b10_LPF.log' ]
    filenames10 = [ 'carv/20160624/10_2_1_LPF.log', 'carv/20160624/10_2_2_LPF.log', 'carv/20160624/10_2_3_LPF.log', 'carv/20160624/10_2_4_LPF.log', 'carv/20160624/10_2_5_LPF.log', 'carv/20160624/10_2_6_LPF.log', 'carv/20160624/10_2_7_LPF.log', 'carv/20160624/10_2_8_LPF.log', 'carv/20160624/10_2_9_LPF.log', 'carv/20160624/10_2_10_LPF.log' ]
    '''


    # b
    filenames1 = [ 'carv/20160624/10_2_10b1_LPF.log', 'carv/20160624/10_2_10b2_LPF.log', 'carv/20160624/10_2_10b3_LPF.log', 'carv/20160624/10_2_10b4_LPF.log', 'carv/20160624/10_2_10b5_LPF.log', 'carv/20160624/10_2_10b6_LPF.log', 'carv/20160624/10_2_10b7_LPF.log', 'carv/20160624/10_2_10b8_LPF.log', 'carv/20160624/10_2_10b9_LPF.log', 'carv/20160624/10_2_10b10_LPF.log' ]
    filenames2 = [ 'carv/20160624/10_2_20b1_LPF.log', 'carv/20160624/10_2_20b2_LPF.log', 'carv/20160624/10_2_20b3_LPF.log', 'carv/20160624/10_2_20b4_LPF.log', 'carv/20160624/10_2_20b5_LPF.log', 'carv/20160624/10_2_20b6_LPF.log', 'carv/20160624/10_2_20b7_LPF.log', 'carv/20160624/10_2_20b8_LPF.log', 'carv/20160624/10_2_20b9_LPF.log', 'carv/20160624/10_2_20b10_LPF.log' ]
    filenames3 = [ 'carv/20160624/10_2_40b1_LPF.log', 'carv/20160624/10_2_40b2_LPF.log', 'carv/20160624/10_2_40b3_LPF.log', 'carv/20160624/10_2_40b4_LPF.log', 'carv/20160624/10_2_40b5_LPF.log', 'carv/20160624/10_2_40b6_LPF.log', 'carv/20160624/10_2_40b7_LPF.log', 'carv/20160624/10_2_40b8_LPF.log', 'carv/20160624/10_2_40b9_LPF.log', 'carv/20160624/10_2_40b10_LPF.log' ]
    filenames4 = [ 'carv/20160624/10_2_80b1_LPF.log', 'carv/20160624/10_2_80b2_LPF.log', 'carv/20160624/10_2_80b3_LPF.log', 'carv/20160624/10_2_80b4_LPF.log', 'carv/20160624/10_2_80b5_LPF.log', 'carv/20160624/10_2_80b6_LPF.log', 'carv/20160624/10_2_80b7_LPF.log', 'carv/20160624/10_2_80b8_LPF.log', 'carv/20160624/10_2_80b9_LPF.log', 'carv/20160624/10_2_80b10_LPF.log' ]
    filenames5 = [ 'carv/20160624/10_2_1_LPF.log', 'carv/20160624/10_2_2_LPF.log', 'carv/20160624/10_2_3_LPF.log', 'carv/20160624/10_2_4_LPF.log', 'carv/20160624/10_2_5_LPF.log', 'carv/20160624/10_2_6_LPF.log', 'carv/20160624/10_2_7_LPF.log', 'carv/20160624/10_2_8_LPF.log', 'carv/20160624/10_2_9_LPF.log', 'carv/20160624/10_2_10_LPF.log' ]
    
    data_list1 = []
    data_list2 = []
    data_list3 = []
    data_list4 = []
    data_list5 = []
    data_list6 = []
    data_list7 = []
    data_list8 = []
    data_list9 = []
    data_list10 = []

    #pres_begin = 0.05
    #pres_end = 0.1
    pres_begin = 20.95
    pres_end = 21.0
    
    for filename in filenames1:
        with open(filename, 'r') as read_file:
            data_list1 += filter.window_function(read_file.readlines(), pres_begin, pres_end)
    for filename in filenames2:
        with open(filename, 'r') as read_file:
            data_list2 += filter.window_function(read_file.readlines(), pres_begin, pres_end)
    for filename in filenames3:
        with open(filename, 'r') as read_file:
            data_list3 += filter.window_function(read_file.readlines(), pres_begin, pres_end)
    for filename in filenames4:
        with open(filename, 'r') as read_file:
            data_list4 += filter.window_function(read_file.readlines(), pres_begin, pres_end)
    for filename in filenames5:
        with open(filename, 'r') as read_file:
            data_list5 += filter.window_function(read_file.readlines(), pres_begin, pres_end)
    '''
    for filename in filenames6:
        with open(filename, 'r') as read_file:
            data_list6 += filter.window_function(read_file.readlines(), pres_begin, pres_end)
    for filename in filenames7:
        with open(filename, 'r') as read_file:
           data_list7 += filter.window_function(read_file.readlines(), pres_begin, pres_end)
    for filename in filenames8:
        with open(filename, 'r') as read_file:
            data_list8 += filter.window_function(read_file.readlines(), pres_begin, pres_end)
    for filename in filenames9:
        with open(filename, 'r') as read_file:
            data_list9 += filter.window_function(read_file.readlines(), pres_begin, pres_end)
    for filename in filenames10:
        with open(filename, 'r') as read_file:
            data_list10 += filter.window_function(read_file.readlines(), pres_begin, pres_end)
    '''

    #print statistics.anova5(data_list1, data_list2, data_list3, data_list4, data_list5)
    print statistics.pairwise_tukeyhsd5(data_list1, data_list2, data_list3, data_list4, data_list5, "tukeyhsd_b21.log")
    #print statistics.pairwise_tukeyhsd10(data_list1, data_list2, data_list3, data_list4, data_list5, data_list6, data_list7, data_list8, data_list9, data_list10, "tukeyhsd_ab21.log")
