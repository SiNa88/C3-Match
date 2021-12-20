import numpy
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter
from scipy.stats.mstats import rankdata    
import networkx as nx
import yaml
import subprocess
import os
import json
import time
import sys

sys.path.append('./diff_times')
from diff_times import comp_times
a = [1,2,3,4,5]
print(a.index(5))
start_time = time.monotonic()

#              AWSVirg Google Exo(lg) Exo(med) EGS  Lenovo  NvJ   RPi4   RPi3
encode_200   = [0.37,   0.37,  0.55,    0.67,  0.17, 0.33,  1.9,  2.16,   2.5] #seconds
encode_1500  = [0.82,   1.005,  0.90,   1.33,  0.36, 0.42, 2.63,  3.19,  7.35] #seconds
encode_3000  = [1.2,     1.5,   1.2,     2.1,  0.47, 0.59, 3.48,   4.4,  8.44] #seconds
encode_6500  = [2.9,     3.7,   2.5,    4.78,  1.22, 1.59, 9.68,  11.8,  22.7] #seconds
encode_20000 = [6.58,    8.8,   5.8,   11.22,   2.7, 3.16, 20.64,   28,    60] #seconds

#              AWSVirg Google Exo(lg) Exo(med) EGS  Lenovo  NvJ   RPi4   RPi3
frame_200   = [46,      11,     7,       6,     1,   1,      1,     2,   2]
frame_1500  = [46,      12,     7,       7,     1,   1,      3,     3,   3]
frame_3000  = [48,      12,     8,       8,     2,   2,      4,     3,   3] 
frame_6500  = [59,      16,    10,      10,     4,   4,     12,     6,   6]
frame_20000 = [78,      20,    12,      14,     6,   6,     27,    24,  24]

# 		                    AWSVirg Google Exo(lg) Exo(med) EGS  Lenovo  NvJ   RPi4   RPi3   
Inference_high_accuracy_model = [0.330, 0.3, 0.290, 0.256, 0.225, 0.282,  1.94, 1.05, 1.5]
Low_accuracy_training_model =   [25,     24,    24,    26,    17,    18,   152,  102, 1000] #seconds
High_accuracy_training_model =  [81,     93,    84,   114,    33,    57,   232,  467, 1000] #seconds


#https://aws.amazon.com/kinesis/data-firehose/pricing/?nc=sn&loc=3

#seg_size = 80000 #(10KB)
#video_size = 8000000000 #(1GB)
index_of_segment = 4
seg_size = [286720, 2457600, 3440640, 14400000, 20971520 ] #bits
video_size = [2000000, 14000000, 28000000, 60000000, 204800000] #bits

tasks0 = sys.argv[1].strip("][").split(",")
#tasks0 = ["encoding","framing","inference","lowAccuracy","highAccuracy","transcoding","packaging"]
##levels = [1,2,3,4,4,5,6]
####print(tasks0)

num_of_apps = 1
newtasks = [0 for i in range(len(tasks0)*num_of_apps)]
for k in range(num_of_apps):
	for i in range(len(tasks0)):
		newtasks[i+(k*len(tasks0))] = tasks0[i]+str(k)
		#print (i," ",(k))


#              0	       1	      2      		3      4        5        6       7		8
#resources = ["vm-aws","vm-googl","vm-exo-lg","vm-exo-med","egs","lenovo","jetson","rpi4","rpi3"]
resources = sys.argv[2].strip("][").split(",")
#101,23,11.5,11.9,0.5,0.5,0.5,0.5
lat = [101e-3,23e-3,11.5e-3,11.9e-3,.5e-3,.5e-3,.5e-3,.5e-3,.5e-3] #ms
#print (len(lat))
'''
46, 46, 48, 59, 78
11, 12, 12, 16, 20
 7,  7,  8, 10, 12
 6,  7,  8, 10, 14
 1,  1,  2,  4,  6
 1,  3,  4, 12, 27
 2,  3,  3,  6, 24
'''
#              AWSVirg Google Exo(lg) Exo(med) EGS  Lenovo  NvJ   RPi4   RPi3
Time_commu_Lenovo_aws = [46, 46, 48, 59, 78] #sec
Time_commu_Lenovo_googl = [11, 12, 12, 16, 20] #sec
Time_commu_Lenovo_exo_lg = [7,  7,  8, 10, 12] #sec
Time_commu_Lenovo_exo_med = [6,  7,  8, 10, 14] #sec
Time_commu_Lenovo_egs = [1,  1,  2,  4,  6] #sec
Time_commu_Lenovo_rpi4 = [1,  3,  4, 12, 27] #sec
Time_commu_Lenovo_njn  = [2,  3,  3,  6, 24] #sec

thrput = [[0.9*8000000 , 2.8*8000000 , 4.2*8000000 , 8.5*8000000 , 11*8000000], #(MB/s) -> (b/s)
		[1.3*8000000 , 10.1*8000000 , 13.5*8000000 , 29*8000000 , 44*8000000], #(MB/s) -> (b/s)
		[2.5*8000000 , 15*8000000 , 25*8000000 , 45*8000000 , 65*8000000], #(MB/s) -> (b/s)
		[2.8*8000000 , 16*8000000 , 25*8000000 , 50*8000000 , 66*8000000], #(MB/s) -> (b/s)
		[25*8000000 , 75*8000000 , 85*8000000 , 95*8000000 , 99*8000000], #(MB/s) -> (b/s)
		[20*8000000 , 40*8000000 , 42*8000000 , 45*8000000 , 46*8000000], #(MB/s) -> (b/s)
		[12*8000000 , 48*8000000 , 50*8000000 , 55*8000000 , 65*8000000]] #(MB/s) -> (b/s)
#print (len(thrput[0]))
'''
thrput_commu_Lenovo_aws = [0.9*8000000 , 2.8*8000000 , 4.2*8000000 , 8.5*8000000 , 11*8000000] #(MB/s) -> (b/s)
thrput_commu_Lenovo_googl = [1.3*8000000 , 10.1*8000000 , 13.5*8000000 , 29*8000000 , 44*8000000] #(MB/s) -> (b/s)
thrput_commu_Lenovo_exo_lg = [2.5*8000000 , 15*8000000 , 25*8000000 , 45*8000000 , 65*8000000] #(MB/s) -> (b/s)
thrput_commu_Lenovo_exo_med = [2.8*8000000 , 16*8000000 , 25*8000000 , 50*8000000 , 66*8000000] #(MB/s) -> (b/s)
thrput_commu_Lenovo_egs = [25*8000000 , 75*8000000 , 85*8000000 , 95*8000000 , 99*8000000] #(MB/s) -> (b/s)
thrput_commu_Lenovo_rpi4 = [20*8000000 , 40*8000000 , 42*8000000 , 45*8000000 , 46*8000000] #(MB/s) -> (b/s)
thrput_commu_Lenovo_njn = [12*8000000 , 48*8000000 , 50*8000000 , 55*8000000 , 65*8000000] #(MB/s) -> (b/s)
'''
SIZE = 208

#      	AWSVirg		   Google	   Exo(lg)	Exo(med)	EGS       lenovo      NvJ       RPi4       RPi3
BW_r = [100000000 , 870000000, 840000000, 840000000, 920000000, 920000000, 450000000 , 800000000 , 328000000]#bps

T = [[[[0] for k in range(len(resources))] for i in range(len(tasks0))] for j in range(num_of_apps)] 
Tm = [[[[0] for k in range(len(resources))] for i in range(len(tasks0))] for j in range(num_of_apps)] 
Tr = [[[[0] for k in range(len(resources))] for i in range(len(tasks0))] for j in range(num_of_apps)] 
Tq = [[[[[0] for j in range(len(resources))] for k in range(10)] for i in range(len(tasks0))] for l in range(num_of_apps)] # Queuing of Data cells.

#T = [[0] * len(resources) for i in range(len(tasks0))]
#Tm = [[0] * len(resources) for i in range(len(tasks0))]
#Tr = [[0] * len(resources) for i in range(len(tasks0))]
#Tq = [[[[0] for j in range(len(resources))] for k in range(10)] for i in range(len(tasks0))] # Queuing of Data cells.

for i in range(num_of_apps):
	Tm[i],Tr[i],Tq[i],T[i] = comp_times()

# Tm[i][:][:],Tr[i][:][:],Tq[i][:][:][:],T[i][:][:]
command = str.encode(os.popen("C:\\Users\\narmehran\\AppData\\Local\\Programs\\Python\\Python39\\python.exe "+"D:\\00Research\\matching\\scheduler\\paper\\ranking.py "+sys.argv[1]+" "+sys.argv[2]).read())
output = command.decode()
#print((output))


candidates_nodes = ((((output))).split(","))[0:len(newtasks)]
for i in range(len(candidates_nodes)):
	candidates_nodes[i]=((((candidates_nodes[i]).replace("\n","")).replace("]","")).replace("[","")).replace(" ","")
#print (candidates_nodes)
alloc = list(map(int, candidates_nodes))
#print (alloc)

#Earliest start time
EST = [[[[0] for j in range(len(resources))] for k in range(10)] for i in range(len(tasks0))]


print ("=====================================================================")
print ("Microservice\t","Resource\t","EarliestStartTime\t", "ProcessTime")
print ("=====================================================================")
for row in range(len(tasks0)):
        if (row != 0):
                EST[row][1][alloc[row]] = EST[row-1][1][alloc[row-1]] + T[0][row-1][alloc[row-1]]
        else:
                EST[row][1][alloc[row]] = 0
        print (tasks0[row],"\t",resources[alloc[row]],"\t",numpy.round(float(EST[row][1][alloc[row]]),2),"\t\t\t", numpy.round(float(T[0][row][alloc[row]]),2))
        print ("--------------------------------------------------------------------")
print ("=====================================================================")
print ("Application completion time: {} second(s)".format(numpy.round(float(EST[len(tasks0)-1][1][alloc[len(tasks0)-1]])+float(T[0][len(tasks0)-1][alloc[len(tasks0)-1]]),2)))
print ("=====================================================================")
elapsed_time = time.monotonic() - start_time
print ("Execution done - time:{}".format(elapsed_time))

EST_coda = [[0] * len(resources) for i in range(len(tasks0))]
alloc_coda = [4, 4, 5, 5, 2]
for row in range(len(tasks0)):
        if (row != 0):
            EST_coda[row][alloc_coda[row]] = EST_coda[row-1][alloc_coda[row-1]] + T[0][row-1][alloc_coda[row-1]]
        else:
            EST_coda[row][alloc_coda[row]] = 0
        print (tasks0[row],"\t",resources[alloc_coda[row]],"\t",numpy.round(float(EST_coda[row][alloc_coda[row]]),2),"\t\t\t", numpy.round(float(T[0][row][alloc_coda[row]]),2))
        #print ((T[row][alloc_sealeap[row]]))
        print ("--------------------------------------------------------------------")
print ("Application completion time on the coda is: {} second(s)".format(numpy.round(float(EST_coda[len(tasks0)-1][alloc_coda[len(tasks0)-1]])+float(T[0][len(tasks0)-1][alloc_coda[len(tasks0)-1]]),2)))


EST_sealeap = [[0] * len(resources) for i in range(len(tasks0))]

#alloc_sealeap = [0,6,4,6,6,4]
#alloc_sealeap = [1,1,1,1,1]
#alloc_sealeap = [5,5,5,5,5]
alloc_sealeap = [2,2,2,2,2]
for row in range(len(tasks0)):
        if (row != 0):
            EST_sealeap[row][alloc_sealeap[row]] = EST_sealeap[row-1][alloc_sealeap[row-1]] + T[0][row-1][alloc_sealeap[row-1]]
        else:
            EST_sealeap[row][alloc_sealeap[row]] = 0
        print (tasks0[row],"\t",resources[alloc_sealeap[row]],"\t",numpy.round(float(EST_sealeap[row][alloc_sealeap[row]]),2),"\t\t\t", numpy.round(float(T[0][row][alloc_sealeap[row]]),2))
        #print ((T[row][alloc_sealeap[row]]))
        print ("--------------------------------------------------------------------")
print ("Application completion time on the sealeap is: {} second(s)".format(numpy.round(float(EST_sealeap[len(tasks0)-1][alloc_sealeap[len(tasks0)-1]])+float(T[0][len(tasks0)-1][alloc_sealeap[len(tasks0)-1]]),2)))


EST_kcss = [[0] * len(resources) for i in range(len(tasks0))]
#alloc_kcss = [4,6,6,6,4]
alloc_kcss = [1,1,1,1,1]
for row in range(len(tasks0)):
        if (row != 0):
            EST_kcss[row][alloc_kcss[row]] = EST_kcss[row-1][alloc_kcss[row-1]] + T[0][row-1][alloc_kcss[row-1]]
        else:
            EST_kcss[row][alloc_kcss[row]] = 0
        print (tasks0[row],"\t",resources[alloc_kcss[row]],"\t",numpy.round(float(EST_kcss[row][alloc_kcss[row]]),2),"\t\t\t", numpy.round(float(T[0][row][alloc_kcss[row]]),2))
        #print ((T[row][alloc_kcss[row]]))
        print ("--------------------------------------------------------------------")
print ("Application completion time on the KCSS is: {} second(s)".format(numpy.round(float(EST_kcss[len(tasks0)-1][alloc_kcss[len(tasks0)-1]])+float(T[0][len(tasks0)-1][alloc_kcss[len(tasks0)-1]]),2)))



EST_aws = [0 for i in range(len(tasks0))]
for row in range(len(tasks0)):
        if (row != 0):
            EST_aws[row] = EST_aws[row-1] + T[0][row-1][0]
        else:
            EST_aws[row] = 0
        print (tasks0[row],"\t",resources[0],"\t",numpy.round(float(EST_aws[row]),2),"\t\t\t", numpy.round(float(T[0][row][0]),2))
        #print ((T[row][alloc[row]]))
        print ("--------------------------------------------------------------------")
print ("Application completion time on the Cloud is: {} second(s)".format(numpy.round(float(EST_aws[len(tasks0)-1])+float(T[0][len(tasks0)-1][0]),2)))
print ("=====================================================================")

#print()
#print()
#print()
#print()

print("------------------TrafficIntensity---------------")
sum_traffic_C3_Match = 0
for i in range(SIZE):
	sum_traffic_C3_Match += (
	(seg_size[index_of_segment]/thrput[4][index_of_segment])
	+(seg_size[index_of_segment]/thrput[2][index_of_segment])
	)
sum_traffic_C3_Match+=(seg_size[index_of_segment]/thrput[2][index_of_segment])
print ("C3_Match: ", sum_traffic_C3_Match/3)#numpy.round(sum_traffic_C3_Match/1024/1024/8,0) , "MB")

sum_traffic_coda = 0
for i in range(SIZE):
	sum_traffic_coda += (
	(seg_size[index_of_segment]/thrput[2][index_of_segment])
	+(seg_size[index_of_segment]/thrput[2][index_of_segment])
	)
sum_traffic_coda+=(video_size[index_of_segment]/thrput[4][index_of_segment])
print ("CODA: ", sum_traffic_coda/3)#numpy.round(sum_traffic_coda/1024/1024/8,0) , "MB")

sum_traffic_sealeap = 0
for i in range(SIZE):
	sum_traffic_sealeap += ((seg_size[index_of_segment]/thrput[2][index_of_segment])
	+(seg_size[index_of_segment]/thrput[2][index_of_segment])
	)
sum_traffic_sealeap+=(video_size[index_of_segment]/thrput[2][index_of_segment])
print ("Sea-leap: ", sum_traffic_sealeap/3)#numpy.round(sum_traffic_kcss/1024/1024/8,0) , "MB")

sum_traffic_kcss = 0
for i in range(SIZE):
	sum_traffic_kcss += ((seg_size[index_of_segment]/thrput[1][index_of_segment])
	+(seg_size[index_of_segment]/thrput[1][index_of_segment])
	)
sum_traffic_kcss+=(video_size[index_of_segment]/thrput[1][index_of_segment])
print ("KCSS: ", sum_traffic_kcss/3)#numpy.round(sum_traffic_kcss/1024/1024/8,0) , "MB")

sum_traffic_cloud = 0
for i in range(SIZE):
	sum_traffic_cloud += ((seg_size[index_of_segment]/thrput[4][index_of_segment])
	+(seg_size[index_of_segment])/thrput[0][index_of_segment])
sum_traffic_cloud+=(video_size[index_of_segment]/thrput[0][index_of_segment])
print ("Heft-cc: ", sum_traffic_cloud/3)#numpy.round(sum_traffic_cloud/1024/1024/8,0) , "MB")


print("----------------------------------------")
print()

'''      
(RPi3B, 1.0)
(RPi4, 0.7)
(Jetson, 0.71)
(Lenovo, 1.25)
(EGS, 1.5)
(ES Medium, 4.1)
(ES Large, 3.9) 
(Google, 5.4)
(AWS m5a.xlarge, 5.4)
'''

tau = [7/180,(5.4/180),(3.9/134),(4.1/192),(1.5/65),(1.25/65),(0.71/432),(0.7/1050),(1/1630)]
# g/s
'''      
	(RPi3B, 1630)
    (RPi4, 1050)
    (Jetson, 432)
    (EGS, 65.2)
    (ES Tiny, 210.2) 
    (ES Medium, 192.1)
    (ES Large, 134.26) 
    (AWS t2.micro, 220.1)
    (AWS c5.large,178.83) 
    (AWS m5a.xlarge,180.3)
'''
'''
print("------------------co2---------------")
sum_co2_C3_Match = 0
for i in range(SIZE):
	sum_co2_C3_Match += (
	#(tau[2]*video_size[index_of_segment]/thrput_commu_Lenovo_exo_lg[index_of_segment])
	(tau[2]*seg_size[index_of_segment]/thrput_commu_Lenovo_exo_lg[index_of_segment])
	+(tau[4]*seg_size[index_of_segment]/thrput_commu_Lenovo_egs[index_of_segment])
	)
sum_co2_C3_Match += (tau[2]*video_size[index_of_segment]/thrput_commu_Lenovo_exo_lg[index_of_segment])
print ("C3_Match: ", sum_co2_C3_Match)#numpy.round(sum_traffic_C3_Match/1024/1024/8,0) , "MB")

sum_co2_coda = 0
for i in range(SIZE):
	sum_co2_coda += (
	(tau[4]*seg_size[index_of_segment]/thrput_commu_Lenovo_exo_lg[index_of_segment])
	+(tau[2]*seg_size[index_of_segment]/thrput_commu_Lenovo_exo_lg[index_of_segment])
	#+(tau[4]*seg_size[index_of_segment]/thrput_commu_Lenovo_egs[index_of_segment])
	#+(tau[2]*seg_size[index_of_segment]/thrput_commu_Lenovo_exo_lg[index_of_segment])
	)
sum_co2_coda +=(tau[4]*video_size[index_of_segment]/thrput_commu_Lenovo_egs[index_of_segment])
#sum_co2_coda+=(tau[2]*seg_size[index_of_segment]/thrput_commu_Lenovo_exo_lg[index_of_segment])#/BW_r[1])
print ("CODA: ", sum_co2_coda)#numpy.round(sum_traffic_coda/1024/1024/8,0) , "MB")

sum_co2_sealeap = 0
for i in range(SIZE):
	sum_co2_sealeap += (tau[2]*(seg_size[index_of_segment]/thrput_commu_Lenovo_exo_lg[index_of_segment])
	+(tau[2]*seg_size[index_of_segment]/thrput_commu_Lenovo_exo_lg[index_of_segment])
	#+(tau[2]*seg_size[index_of_segment]/thrput_commu_Lenovo_exo_lg[index_of_segment])
	#+(tau[3]*seg_size[index_of_segment]/thrput_commu_Lenovo_exo_med[index_of_segment])
	#+(tau[2]*seg_size[index_of_segment]/thrput_commu_Lenovo_exo_lg[index_of_segment])
	#+(tau[2]*seg_size[index_of_segment]/thrput_commu_Lenovo_exo_lg[index_of_segment])
	)
sum_co2_sealeap+=(tau[2]*video_size[index_of_segment]/thrput_commu_Lenovo_exo_lg[index_of_segment])#/BW_r[2])
print ("Sea-leap: ", sum_co2_sealeap)#numpy.round(sum_co2_kcss/1024/1024/8,0) , "MB")

#alloc_kcss = [1,1,1,3,3,3]
sum_co2_kcss = 0
for i in range(SIZE):
	sum_co2_kcss += (tau[1]*(seg_size[index_of_segment]/thrput_commu_Lenovo_googl[index_of_segment])
	+(tau[1]*seg_size[index_of_segment]/thrput_commu_Lenovo_googl[index_of_segment])
	#+(seg_size[index_of_segment]/thrput_commu_Lenovo_njn[index_of_segment])
	#+(seg_size[index_of_segment]/thrput_commu_Lenovo_egs[index_of_segment])
	#+(seg_size[index_of_segment]/thrput_commu_Lenovo_njn[index_of_segment])
	#+(tau[1]*seg_size[index_of_segment]/thrput_commu_Lenovo_googl[index_of_segment])
	)
sum_co2_kcss+=(tau[1]*video_size[index_of_segment]/thrput_commu_Lenovo_googl[index_of_segment])#/BW_r[2])
print ("KCSS: ", sum_co2_kcss)#numpy.round(sum_co2_kcss/1024/1024/8,0) , "MB")


sum_co2_cloud = 0
for i in range(SIZE):
	sum_co2_cloud += (tau[0]*(seg_size[index_of_segment]/thrput_commu_Lenovo_egs[index_of_segment])
	#+(seg_size[index_of_segment]/thrput_commu_Lenovo_aws[index_of_segment])
	#+(seg_size[index_of_segment]/thrput_commu_Lenovo_aws[index_of_segment])
	+(tau[0]*seg_size[index_of_segment]/thrput_commu_Lenovo_aws[index_of_segment]))
sum_co2_cloud+=(tau[0]*video_size[index_of_segment]/thrput_commu_Lenovo_aws[index_of_segment])#BW_r[0])
print ("Heft-cc: ", sum_co2_cloud)#numpy.round(sum_co2_cloud/1024/1024/8,0) , "MB")


print("----------------------------------------")
print()'''

print("------------------co2---------------")
sum_co2_C3_Match = 0

sum_co2_C3_Match += ((tau[2]+tau[4]+tau[2])*(numpy.round(float(T[0][len(tasks0)-1][alloc[len(tasks0)-1]]),2)))
print ("C3_Match: ", sum_co2_C3_Match)

sum_co2_coda = 0

sum_co2_coda +=((tau[4]+tau[2]+tau[4])*(numpy.round(float(T[0][len(tasks0)-1][alloc_coda[len(tasks0)-1]]),2)))
print ("CODA: ", sum_co2_coda)

sum_co2_sealeap = 0
sum_co2_sealeap+=(tau[2]*3*numpy.round(float(T[0][len(tasks0)-1][alloc_sealeap[len(tasks0)-1]]),2))
print ("Sea-leap: ", sum_co2_sealeap)

sum_co2_kcss = 0

sum_co2_kcss+=(tau[1]*3*numpy.round(float(T[0][len(tasks0)-1][alloc_kcss[len(tasks0)-1]]),2))
print ("KCSS: ", sum_co2_kcss)


sum_co2_cloud = 0
sum_co2_cloud+=(tau[0]*3*numpy.round(float(T[0][len(tasks0)-1][0]),2))
print ("Heft-cc: ", sum_co2_cloud)


print("----------------------------------------")
print()
