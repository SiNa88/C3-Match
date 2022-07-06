import re
import numpy
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter
from scipy.stats.mstats import rankdata    
import networkx as nx
import yaml
import json
import subprocess
import os
import json
import time
import sys

sys.path.append('./diff_times')
from diff_times import comp_times


start_time = time.time_ns()#monotonic()

#seg_size = 80000 #(10KB)
#video_size = 8000000000 #(1GB)
index_of_segment = 4
#SIZE = 1
#seg_size = [286720, 2457600, 3440640, 14400000, 20971520 ] #bits
#video_size = [2000000, 14000000, 28000000, 60000000, 204800000] #bits

# file_size = 5774209850540032 # 688(MB) #compressed one
seg_size = [3.2*1024*1024*8, 3.5*1024*1024*8, 3.8*1024*1024*8, 4.1*1024*1024*8, 4.4*1024*1024*8] #bits
#print([3.2*1024, 3.5*1024, 3.8*1024, 4.1*1024, 4.4*1024])
SIZE = 10
app = [{'id':0, 'tasks':['encode_20000', 'frame_20000', 'hightrain', 'inference', 'package']}, {'id':1, 'tasks':['download_time', 'extract_data', 'train_runtime_83', 'train_runtime_86', 'package2']}]


num_of_apps = len(sys.argv) - 2
lenn_app = 5 
levels = [[0  for i in range(lenn_app)] for j in range(num_of_apps)]
for i in range(1,len(sys.argv)-1):
	levels[i-1] = sys.argv[i].strip("][").split(",")
	#print (levels[i-1])

#              0	       1	      2      		3      4        5        6       7		8
#resources = ["vm-bg","vm-exo-med","vm-exo-lg","egs","lenovo","jetson","rpi4","rpi3"] #["vm-aws","vm-googl","vm-exo-lg","vm-exo-med","egs","lenovo","jetson","rpi4","rpi3"]
resources = sys.argv[len(sys.argv)-1].strip("][").split(",")
#print (levels)



#              0	       1	      2      		3      4        5        6       7
#BG, FRA VIE Gateway Lenovo NANO  RPI4   RPI3



lat =	[[ 0.0001e-3, 26e-3, 15e-3 , 22.4e-3 , 22.8e-3 , 22.8e-3 , 22.8e-3 , 22.8e-3],
		[ 26e-3,  0.0001e-3 , 12.5e-3 , 18.4e-3 , 18.4e-3 , 18.4e-3 , 18.4e-3 , 18.4e-3],
		[ 15e-3 , 12.5e-3 , 0.0001e-3  , 7.2e-3 , 7.2e-3 , 7.5e-3 , 7.5e-3 , 7.5e-3], 
		[ 22.4e-3, 18e-3 , 7.2e-3,  0.0001e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3], 
		[ 22.8e-3, 18.4e-3, 7.2e-3, 0.5e-3 , 0.0001e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3], 
		[ 22.8e-3, 18.4e-3, 7.5e-3, 0.5e-3 , 0.5e-3 , 0.0001e-3 , 0.5e-3 , 0.5e-3],  
		[ 22.8e-3, 18.4e-3 , 7.5e-3, 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.0001e-3 , 0.5e-3], 
		[ 22.8e-3, 18.4e-3 , 7.5e-3, 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.0001e-3]]


BW_r = 	[[ 1000 , 0.76,   1.5 , 0.92 , 0.9 , 0.9 , 0.85 , 0.4],
		[ 0.76 ,  1000 , 1.6 , 0.93 , 0.85 , 0.7 , 0.77, 0.4],
		[ 1.5 , 1.6 , 1000  , 0.95 , 0.9 , 0.9 , 0.9 ,0.4], 
		[ 0.92 , 0.93 , 0.95,  1000  , 0.86 ,  0.93 , 0.85 ,0.4], 
		[ 0.9 , 0.85, 0.9, 0.86,1000, 0.92, 0.85 ,0.4 ], 
		[ 0.9, 0.7, 0.9, 0.93, 0.92 , 1000, 0.88 ,0.4],  
		[ 0.85, 0.77, 0.9, 0.85, 0.85, 0.88, 1000, 0.4 ], 
		[0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,  1000]]

T  = [[[[0] for k in range(len(resources))] for i in range(lenn_app)] for j in range(num_of_apps)] 
Tm = [[[[0] for k in range(len(resources))] for i in range(lenn_app)] for j in range(num_of_apps)] 
Tr = [[[[0] for k in range(len(resources))] for i in range(lenn_app)] for j in range(num_of_apps)] 
Tq = [[[[0] for k in range(len(resources))] for i in range(lenn_app)] for j in range(num_of_apps)]  #[[[[[0] for j in range(len(resources))] for k in range(10)] for i in range(lenn_app)] for l in range(num_of_apps)] # Queuing of Data cells.


command = str.encode(os.popen("C:\\Users\\narmehran\\AppData\\Local\\Programs\\Python\\Python39\\python.exe "+"D:\\00Research\\matching\\scheduler\\paper\\MoreApps\\ranking-levelized.py "+sys.argv[1]+" "+sys.argv[2]+" "+sys.argv[3]+" "+sys.argv[4]+" "+sys.argv[5]+" "+sys.argv[6]+" "+sys.argv[7]+" "+sys.argv[8]+" "+sys.argv[9]+" "+sys.argv[10]+" "+sys.argv[11]+" "+sys.argv[12]+" "+sys.argv[13]+" "+sys.argv[14]+" "+sys.argv[15]+" "+sys.argv[16]+" "+sys.argv[17]+" "+sys.argv[18]+" "+sys.argv[19]+" "+sys.argv[20]+" "+sys.argv[21]+" "+sys.argv[22]+" "+sys.argv[23]+" "+sys.argv[24]+" "+sys.argv[25]+" "+sys.argv[26]+" "+sys.argv[27]+" "+sys.argv[28]+" "+sys.argv[29]+" "+sys.argv[30]+" "+" "+sys.argv[31]+" "+sys.argv[32]+" "+sys.argv[33]+" "+sys.argv[34]+" "+sys.argv[35]+" "+sys.argv[36]+" "+sys.argv[37]+" "+sys.argv[38]+" "+sys.argv[39]+" "+sys.argv[40]+" "+sys.argv[41]).read())#).read())#
output = command.decode()
#print((output))
elapsed_time = time.time_ns() - start_time
print ("Execution done - time:{}".format(elapsed_time))
print ("=====================================================================")
print ("=====================================================================")
print ("=====================================================================")

candidates_nodes = ((((output))).split(","))[0:lenn_app*num_of_apps]
for i in range(len(candidates_nodes)):
	candidates_nodes[i]=((((candidates_nodes[i]).replace("\n","")).replace("]","")).replace("[","")).replace(" ","")
#print (candidates_nodes)
alloc = list(map(int, candidates_nodes))
#############print ((alloc))

for k in range(num_of_apps):
	for i in range(lenn_app):
		if (i != 0):
			#print(alloc[i+(k*lenn_app)-1])
			Tm[k][i],Tr[k][i],Tq[k][i],T[k][i] = comp_times(resources,levels[k][i],alloc[i+(k*lenn_app)-1]) #comp_times(levels[i],alloc[i-1]) #
		else:
			Tm[k][i],Tr[k][i],Tq[k][i],T[k][i] = comp_times(resources,levels[k][i],4) # comp_times(levels[i],5) #

#Earliest start time
EST = [[[[0] for k in range(len(resources))] for i in range(lenn_app)] for j in range(num_of_apps)]


print ("=====================================================================")
print ("Microservice\t","Resource\t","EarliestStartTime\t", "ProcessTime")
print ("=====================================================================")
sum = 0
sum_qp = 0
sum_c = 0
for k in range(num_of_apps):
	for row in range(lenn_app):
		if (row != 0):
			EST[k][row][alloc[(row+(k*lenn_app))]] = EST[k][row-1][alloc[(row+(k*lenn_app))-1]] + T[k][row-1][alloc[(row+(k*lenn_app))-1]]
		else:
			EST[k][row][alloc[(row+(k*lenn_app))]] = 0
		#print (levels[k][row],"\t",resources[alloc[(row+(k*lenn_app))]],"\t",numpy.round(float(EST[k][row][alloc[(row+(k*lenn_app))]]),2),"\t\t\t", numpy.round(float(T[k][row][alloc[(row+(k*lenn_app))]]),2))
		##########print(k,"   ",numpy.round(Tm[k][row][alloc[(row+(k*lenn_app))]],2),"   ",numpy.round(Tq[k][row][alloc[(row+(k*lenn_app))]],2),"   ",numpy.round(Tr[k][row][alloc[(row+(k*lenn_app))]],2),"    ",numpy.round(float(EST[k][row][alloc[(row+(k*lenn_app))]]),2))
		sum_qp += numpy.round(Tm[k][row][alloc[(row+(k*lenn_app))]],2)+numpy.round(Tq[k][row][alloc[(row+(k*lenn_app))]],2)
		sum_c += numpy.round(float(EST[k][row][alloc[(row+(k*lenn_app))]]),2)
		#print ("--------------------------------------------------------------------")
	#print ("=====================================================================")
	print ("{}".format(numpy.round(float(EST[k][lenn_app-1][alloc[(lenn_app+(k*lenn_app))-1]])+float(T[k][lenn_app-1][alloc[(lenn_app+(k*lenn_app))-1]]),2)))
	#print ("=====================================================================")
	sum += numpy.round(float(EST[k][lenn_app-1][alloc[(lenn_app+(k*lenn_app))-1]])+float(T[k][lenn_app-1][alloc[(lenn_app+(k*lenn_app))-1]]),2)
	##########print(numpy.round(sum/(k+1),4))#k,"   ",
#print(sum)
#print(numpy.round(sum/(num_of_apps+1),4))

#print ("=====================================================================")
print ((alloc))
sum = 0
EST_nan = [[[[0] for k in range(len(resources))] for i in range(lenn_app)] for j in range(num_of_apps)]
alloc_nan = [3, 2, 3, 2, 4,
 1, 4, 0, 1, 5,
 3, 2, 3, 2, 4,
 1, 4, 0, 1, 5,
 3, 2, 3, 2, 4,
 1, 4, 0, 1, 5,
 3, 2, 3, 2, 4,
 1, 4, 0, 1, 5,
 3, 2, 3, 2, 4,
 1, 4, 0, 1, 5,
 3, 2, 3, 2, 4,
 1, 4, 0, 1, 5,
 3, 2, 3, 2, 4,
 1, 4, 0, 1, 6,
 2, 3, 1, 3, 2,
 1, 4, 0, 1, 6,
 2, 3, 1, 3, 2,
 1, 4, 0, 1, 6,
 2, 3, 1, 3, 2,
 1, 4, 0, 1, 6,
 3, 2, 3, 2, 4,
 1, 4, 0, 1, 5,
 3, 2, 3, 2, 4,
 1, 4, 0, 1, 6,
 2, 3, 1, 3, 2,
 1, 4, 0, 1, 6,
 2, 3, 1, 3, 2,
 1, 4, 0, 1, 6,
 2, 3, 1, 3, 2,
 1, 4, 0, 1, 6,
 3, 2, 3, 2, 4,
 1, 4, 0, 1, 5,
 3, 2, 3, 2, 4,
 1, 4, 0, 1, 6,
 2, 3, 1, 3, 2,
 1, 4, 0, 1, 6,
 2, 3, 1, 3, 2,
 1, 4, 0, 1, 6,
 2, 3, 1, 3, 2,
 1, 4, 0, 1, 6]
sum_qp = 0
sum_c = 0
for k in range(num_of_apps):
	for row in range(lenn_app):
		if (row != 0):
			EST_nan[k][row][alloc_nan[(row+(k*lenn_app))]] = EST_nan[k][row-1][alloc_nan[(row+(k*lenn_app))-1]] + T[k][row-1][alloc_nan[(row+(k*lenn_app))-1]]
		else:
			EST_nan[k][row][alloc_nan[(row+(k*lenn_app))]] = 0
		#print (levels[k][row],"\t",resources[alloc_nan[(row+(k*lenn_app))]],"\t",numpy.round(float(EST_nan[k][row][alloc_nan[(row+(k*lenn_app))]]),2),"\t\t\t", numpy.round(float(T[k][row][alloc_nan[(row+(k*lenn_app))]]),2))
		##########print(k,"   ",numpy.round(Tm[k][row][alloc_nan[(row+(k*lenn_app))]],2),"   ",numpy.round(Tq[k][row][alloc_nan[(row+(k*lenn_app))]],2),"   ",numpy.round(Tr[k][row][alloc_nan[(row+(k*lenn_app))]],2),"    ",numpy.round(float(EST_nan[k][row][alloc_nan[(row+(k*lenn_app))]]),2))
		#print ("--------------------------------------------------------------------")
	#print ("=====================================================================")
	print ("{}".format(numpy.round(float(EST_nan[k][lenn_app-1][alloc_nan[(lenn_app+(k*lenn_app))-1]])+float(T[k][lenn_app-1][alloc_nan[(lenn_app+(k*lenn_app))-1]]),2)))
	#print ("=====================================================================")
	sum += numpy.round(float(EST_nan[k][lenn_app-1][alloc_nan[(lenn_app+(k*lenn_app))-1]])+float(T[k][lenn_app-1][alloc_nan[(lenn_app+(k*lenn_app))-1]]),2)
	##########print(numpy.round(sum/(k+1),4))#k,"   ",
#print(sum)
#print(numpy.round(sum/(num_of_apps+1),4))
print ((alloc_nan))
sum = 0
EST_sealeap = [[[[0] for k in range(len(resources))] for i in range(lenn_app)] for j in range(num_of_apps)]
alloc_sealeap = [1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1]
sum_qp = 0
sum_c = 0
for k in range(num_of_apps):
	for row in range(lenn_app):
		if (row != 0):
			EST_sealeap[k][row][alloc_sealeap[(row+(k*lenn_app))]] = EST_sealeap[k][row-1][alloc_sealeap[(row+(k*lenn_app))-1]] + T[k][row-1][alloc_sealeap[(row+(k*lenn_app))-1]]
		else:
			EST_sealeap[k][row][alloc_sealeap[(row+(k*lenn_app))]] = 0
		##########print(k,"   ",numpy.round(Tm[k][row][alloc_sealeap[(row+(k*lenn_app))]],2),"   ",numpy.round(Tq[k][row][alloc_sealeap[(row+(k*lenn_app))]],2),"   ",numpy.round(Tr[k][row][alloc_sealeap[(row+(k*lenn_app))]],2),"    ",numpy.round(float(EST_sealeap[k][row][alloc_sealeap[(row+(k*lenn_app))]]),2))
		#print (levels[k][row],"\t",resources[alloc_sealeap[(row+(k*lenn_app))]],"\t",numpy.round(float(EST_sealeap[k][row][alloc_sealeap[(row+(k*lenn_app))]]),2),"\t\t\t", numpy.round(float(T[k][row][alloc_sealeap[(row+(k*lenn_app))]]),2))
		#print ("--------------------------------------------------------------------")
	#print ("=====================================================================")
	print ("{}".format(numpy.round(float(EST_sealeap[k][lenn_app-1][alloc_sealeap[(lenn_app+(k*lenn_app))-1]])+float(T[k][lenn_app-1][alloc_sealeap[(lenn_app+(k*lenn_app))-1]]),2)))
	#print ("=====================================================================")
	sum += numpy.round(float(EST_sealeap[k][lenn_app-1][alloc_sealeap[(lenn_app+(k*lenn_app))-1]])+float(T[k][lenn_app-1][alloc_sealeap[(lenn_app+(k*lenn_app))-1]]),2)
	##########print(numpy.round(sum/(k+1),4)) #k,"   ",
#print(sum)
print(numpy.round(sum/(num_of_apps+1),4))
print ((alloc_sealeap))
EST_kcss = [[[[0] for k in range(len(resources))] for i in range(lenn_app)] for j in range(num_of_apps)]

alloc_kcss = [1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1, 2, 1, 2, 1, 1,2,1,2,1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1 ,1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,2,1,2,1, 1,1,1,2,1, 1,1,1,2,1, 1,1,1,2,1, 1,1,1,2,1, 1,1,1,2,1, 1,1,1,2,1]#0, 4, 0, 0, 0, 2, 0, 1, 0, 2, 0, 2, 2, 1, 4, 5, 2, 5, 2, 5, 6, 3, 0, 6, 6, 3, 2, 3, 2, 1, 2, 1, 3, 2, 1, 2, 1, 3, 2, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 3, 2, 3, 2, 0, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 0, 2, 3, 1, 3, 1, 2, 3, 1, 3, 0, 2, 3, 1, 3, 1, 2, 3, 1, 3, 0]#[3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 0, 4, 0, 0, 0, 2, 0, 1, 0, 3, 0, 3, 2, 1, 4, 5, 2, 5, 2, 5, 6, 3, 0, 6, 6, 3, 2, 3, 2, 1, 2, 1, 3, 2, 1, 2, 1, 3, 2, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 3, 2, 3, 2, 0, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 0, 2, 3, 1, 3, 1, 2, 3, 1, 3, 0, 2, 3, 1, 3, 1, 2, 3, 1, 3, 0]#[3,3,3,3,3,2,2,2,2,2,0, 4, 0, 0, 0, 2, 0, 1, 0, 3, 0, 3, 2, 1, 4, 5, 2, 5, 2, 5, 6, 3, 0, 6, 6, 3, 2, 3, 2, 1, 2, 1, 3, 2, 1, 2, 1, 3, 2, 1, 2, 3, 0, 3, 1, 2, 3, 0, 3, 1, 2, 3, 0, 3, 1, 2, 3, 0, 3, 1, 3, 2, 3, 2, 0, 2, 3, 0, 3, 1, 2, 3, 0, 3, 1, 2, 3, 0, 3, 1, 2, 3, 0, 3, 1, 2, 3, 0, 3, 1, 2, 3, 0, 3, 0, 2, 3, 0, 3, 1, 2, 3, 0, 3, 0, 2, 3, 0, 3, 1, 2, 3, 0, 3, 0]#3, 2, 3, 3, 0, 3, 2, 3, 3, 0, 3, 2, 3, 3, 0, 2, 3, 0, 3, 0, 2, 3, 0, 3, 0, 3, 2, 3, 2, 1, 3, 2, 3, 2, 1, 2, 3, 0, 3, 0, 2, 3, 0, 3, 0, 2, 3, 0, 3, 0]
#print ((alloc_kcss))
sum = 0
sum_qp = 0
sum_c = 0
for k in range(num_of_apps):
	for row in range(lenn_app):
		if (row != 0):
			EST_kcss[k][row][alloc_kcss[(row+(k*lenn_app))]] = EST_kcss[k][row-1][alloc_kcss[(row+(k*lenn_app))-1]] + T[k][row-1][alloc_kcss[(row+(k*lenn_app))-1]]
		else:
			EST_kcss[k][row][alloc_kcss[(row+(k*lenn_app))]] = 0
		##########print(k,"   ",numpy.round(Tm[k][row][alloc_kcss[(row+(k*lenn_app))]],2),"   ",numpy.round(Tq[k][row][alloc_kcss[(row+(k*lenn_app))]],2),"   ",numpy.round(Tr[k][row][alloc_kcss[(row+(k*lenn_app))]],2),"    ",numpy.round(float(EST_kcss[k][row][alloc_kcss[(row+(k*lenn_app))]]),2))
		#print (levels[k][row],"\t",resources[alloc_kcss[(row+(k*lenn_app))]],"\t",numpy.round(float(EST_kcss[k][row][alloc_kcss[(row+(k*lenn_app))]]),2),"\t\t\t", numpy.round(float(T[k][row][alloc_kcss[(row+(k*lenn_app))]]),2))
		#print ("--------------------------------------------------------------------")
	#print ("=====================================================================")
	print ("{}".format(numpy.round(float(EST_kcss[k][lenn_app-1][alloc_kcss[(lenn_app+(k*lenn_app))-1]])+float(T[k][lenn_app-1][alloc_kcss[(lenn_app+(k*lenn_app))-1]]),2)))
	#print ("=====================================================================")
	sum += numpy.round(float(EST_kcss[k][lenn_app-1][alloc_kcss[(lenn_app+(k*lenn_app))-1]])+float(T[k][lenn_app-1][alloc_kcss[(lenn_app+(k*lenn_app))-1]]),2)
	##########print(numpy.round(sum/(k+1),4))#k,"   ",
	#print()
#print(sum)
print ((alloc_kcss))
sum_com_C3_Match = 0
sum_com_nan = 0
sum_com_sealeap = 0
sum_com_kcss = 0
print("------------------------Time_[QP]-------------------------")
for k in range(num_of_apps):
	for row in range(lenn_app):
		sum_com_C3_Match += Tm[k][row-1][alloc[(row+(k*lenn_app))-1]]+Tq[k][row-1][alloc[(row+(k*lenn_app))-1]]
		sum_com_nan += Tm[k][row-1][alloc_nan[(row+(k*lenn_app))-1]]+Tq[k][row-1][alloc_nan[(row+(k*lenn_app))-1]]
		sum_com_sealeap += Tm[k][row-1][alloc_sealeap[(row+(k*lenn_app))-1]]+Tq[k][row-1][alloc_sealeap[(row+(k*lenn_app))-1]]
		sum_com_kcss += Tm[k][row-1][alloc_kcss[(row+(k*lenn_app))-1]]+Tq[k][row-1][alloc_kcss[(row+(k*lenn_app))-1]]
	if(k==9 or k==19 or k==29 or k==39 ):	
		print(numpy.round(sum_com_C3_Match/(k*lenn_app),4)," ",numpy.round(sum_com_nan/(k*lenn_app),4)," ",numpy.round(sum_com_sealeap/(k*lenn_app),4)," ",numpy.round(sum_com_kcss/(k*lenn_app),4))
	#print()