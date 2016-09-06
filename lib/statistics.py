from scipy import stats
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def anova5(list1, list2, list3, list4, list5):
    list1=np.array(list1).T[2]
    list2=np.array(list2).T[2]
    list3=np.array(list3).T[2]
    list4=np.array(list4).T[2]
    list5=np.array(list5).T[2]

    statistic, pvalue = stats.f_oneway(list1, list2, list3, list4, list5)
    
    print statistic, pvalue

def pairwise_tukeyhsd5(list1, list2, list3, list4, list5, filename):
    list1=np.array(list1).T[2]
    list2=np.array(list2).T[2]
    list3=np.array(list3).T[2]
    list4=np.array(list4).T[2]
    list5=np.array(list5).T[2]

    list_all = []
    for value in list1:
        list_all.append([ '10', float(value) ])
    for value in list2:
        list_all.append([ '20', float(value) ])
    for value in list3:
        list_all.append([ '40', float(value) ])
    for value in list4:
        list_all.append([ '80', float(value) ])
    for value in list5:
        list_all.append([ 'inf.', float(value) ])

    ave1 = np.average(list1)
    ave2 = np.average(list2)
    ave3 = np.average(list3)
    ave4 = np.average(list4)
    ave5 = np.average(list5)
    std1 = np.std(list1)
    std2 = np.std(list2)
    std3 = np.std(list3)
    std4 = np.std(list4)
    std5 = np.std(list5)
    print "---------------------------------------------------------------"
    print "         data               average               std          "
    print "---------------------------------------------------------------"
    print "         list1           "+str(ave1)+"       "+str(std1)+"     "
    print "         list2           "+str(ave2)+"       "+str(std2)+"     "
    print "         list3           "+str(ave3)+"       "+str(std3)+"     "
    print "         list4           "+str(ave4)+"       "+str(std4)+"     "
    print "         list5           "+str(ave5)+"       "+str(std5)+"     "
    print "---------------------------------------------------------------\n"
    list_all = np.rec.array(list_all, dtype=[('carv', '|S4'), ('value', float)])
    print pairwise_tukeyhsd(list_all['value'], list_all['carv'])

    with open(filename, 'w') as write_file:
        write_file.write("1 "+str(ave1)+" "+str(std1)+"\n")
        write_file.write("2 "+str(ave2)+" "+str(std2)+"\n")
        write_file.write("3 "+str(ave3)+" "+str(std3)+"\n")
        write_file.write("4 "+str(ave4)+" "+str(std4)+"\n")
        write_file.write("5 "+str(ave5)+" "+str(std5)+"\n")

def pairwise_tukeyhsd10(list1, list2, list3, list4, list5, list6, list7, list8, list9, list10, filename):
    list1=np.array(list1).T[2]
    list2=np.array(list2).T[2]
    list3=np.array(list3).T[2]
    list4=np.array(list4).T[2]
    list5=np.array(list5).T[2]
    list6=np.array(list6).T[2]
    list7=np.array(list7).T[2]
    list8=np.array(list8).T[2]
    list9=np.array(list9).T[2]
    list10=np.array(list10).T[2]

    list_all = []
    for value in list1:
        list_all.append([ 'A10', float(value) ])
    for value in list2:
        list_all.append([ 'A20', float(value) ])
    for value in list3:
        list_all.append([ 'A40', float(value) ])
    for value in list4:
        list_all.append([ 'A80', float(value) ])
    for value in list5:
        list_all.append([ 'Ainf.', float(value) ])
    for value in list6:
        list_all.append([ 'B10', float(value) ])
    for value in list7:
        list_all.append([ 'B20', float(value) ])
    for value in list8:
        list_all.append([ 'B40', float(value) ])
    for value in list9:
        list_all.append([ 'B80', float(value) ])
    for value in list10:
        list_all.append([ 'Binf.', float(value) ])

    ave1 = np.average(list1)
    ave2 = np.average(list2)
    ave3 = np.average(list3)
    ave4 = np.average(list4)
    ave5 = np.average(list5)
    std1 = np.std(list1)
    std2 = np.std(list2)
    std3 = np.std(list3)
    std4 = np.std(list4)
    std5 = np.std(list5)

    ave6 = np.average(list6)
    ave7 = np.average(list7)
    ave8 = np.average(list8)
    ave9 = np.average(list9)
    ave10 = np.average(list10)
    std6 = np.std(list6)
    std7 = np.std(list7)
    std8 = np.std(list8)
    std9 = np.std(list9)
    std10 = np.std(list10)

    print "---------------------------------------------------------------"
    print "         data               average               std          "
    print "---------------------------------------------------------------"
    print "         list1           "+str(ave1)+"       "+str(std1)+"     "
    print "         list2           "+str(ave2)+"       "+str(std2)+"     "
    print "         list3           "+str(ave3)+"       "+str(std3)+"     "
    print "         list4           "+str(ave4)+"       "+str(std4)+"     "
    print "         list5           "+str(ave5)+"       "+str(std5)+"     "
    print "         list6           "+str(ave6)+"       "+str(std6)+"     "
    print "         list7           "+str(ave7)+"       "+str(std7)+"     "
    print "         list8           "+str(ave8)+"       "+str(std8)+"     "
    print "         list9           "+str(ave9)+"       "+str(std9)+"     "
    print "         list10           "+str(ave10)+"       "+str(std10)+"     "
    print "---------------------------------------------------------------\n"
    list_all = np.rec.array(list_all, dtype=[('carv', '|S4'), ('value', float)])
    print pairwise_tukeyhsd(list_all['value'], list_all['carv'])

    with open(filename, 'w') as write_file:
        write_file.write("1 "+str(ave1)+" "+str(std1)+"\n")
        write_file.write("2 "+str(ave2)+" "+str(std2)+"\n")
        write_file.write("3 "+str(ave3)+" "+str(std3)+"\n")
        write_file.write("4 "+str(ave4)+" "+str(std4)+"\n")
        write_file.write("5 "+str(ave5)+" "+str(std5)+"\n")
        write_file.write("6 "+str(ave6)+" "+str(std6)+"\n")
        write_file.write("7 "+str(ave7)+" "+str(std7)+"\n")
        write_file.write("8 "+str(ave8)+" "+str(std8)+"\n")
        write_file.write("9 "+str(ave9)+" "+str(std9)+"\n")
        write_file.write("10 "+str(ave10)+" "+str(std10)+"\n")
