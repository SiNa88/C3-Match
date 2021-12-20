import numpy
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter
from scipy.stats.mstats import rankdata    
import networkx as nx
import yaml
import subprocess
import json
import sys

sys.path.append('./diff_times')
from diff_times import comp_times

'''
from matplotlib import pyplot as plt
g1 = nx.DiGraph()
g1.add_edges_from([("root", "a"), ("a", "b"), ("a", "e"), ("b", "c"), ("b", "d"), ("d", "e")])
plt.tight_layout()
nx.draw_networkx(g1, arrows=True)
plt.savefig("g1.png", format="PNG")
# tell matplotlib you're done with the plot: https://stackoverflow.com/questions/741877/how-do-i-tell-matplotlib-that-i-am-done-with-a-plot
plt.clf()
'''

'''tasksg = nx.DiGraph()
tasksg.add_edges_from([("src","encoding"),("encoding","framing"),("framing","lowAccuracy"),("lowAccuracy","highAccuracy"),("lowAccuracy","analysis"),("highAccuracy","analysis"),("analysis","transcoding"),("transcoding","packaging"),("analysis","packaging"),("packaging","snk")])
plt.tight_layout()
nx.draw_networkx(tasksg, arrows=True)
plt.savefig("g1.png", format="PNG")
plt.clf()
l = list(nx.topological_sort(tasksg))'''
#print(l)
###print(tasksg.edges())
###print(tasksg.nodes())
###print()
###print()

###len_l = len(l)
####print (len_l)
#for i in range(len_l):
#	print("Degree"," ",tasksg.degree[l[i]])
#	print(list(tasksg.predecessors(l[i])))
#	print("In-Degree"," ",tasksg.in_degree(l[i]))
#	print("Out-Degree"," ",tasksg.out_degree(l[i]))
#	print(list(tasksg.successors(l[i])))
#	print()
#	print()


#B = nx.dag_to_branching(tasksg)
#print(B.nodes())
#nx.draw_networkx(B, arrows=True)
#plt.savefig("g2.png", format="PNG")
#plt.clf()

####tasks0 = ["src","framing","lowAccuracy","highAccuracy","analysis","transcoding","packaging","snk"]
####levels = [1,2,3,4,5,6]
####print(tasks0)

# Converting string to list
tasks0 = sys.argv[1].strip("][").split(",")
#print(type(tasks0))

num_of_apps = 1
newtasks = [0 for i in range(len(tasks0)*num_of_apps)]
for k in range(num_of_apps):
	for i in range(len(tasks0)):
		newtasks[i+(k*len(tasks0))] = tasks0[i]+str(k)


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



#              0	       1	      2      		3      4        5        6       7		8
#resources = ["vm-aws","vm-googl","vm-exo-lg","vm-exo-med","egs","lenovo","jetson","rpi4","rpi3"]
resources = sys.argv[2].strip("][").split(",")

#101,23,11.5,11.9,0.5,0.5,0.5,0.5
lat = [101e-3,23e-3,11.5e-3,11.9e-3,.5e-3,.5e-3,.5e-3,.5e-3,.5e-3] #ms

#https://aws.amazon.com/kinesis/data-firehose/pricing/?nc=sn&loc=3

#seg_size = 80000 #(10KB)
#video_size = 8000000000 #(1GB)
index_of_segment = 4
seg_size = [286720, 2457600, 3440640, 14400000, 20971520 ] #bits
video_size = [2000000, 14000000, 28000000, 60000000, 204800000] #bits

#For example, if we assume each cell is 8 bytes and the average record size is 32 columns then the average record size is 256 bytes. A file with 10M records equates to roughly 2.56 GB
#Cell Estimate: 8 bytes/cell * 32 columns * 10,000,000 records = 2.56 GB
#https://rstudio-pubs-static.s3.amazonaws.com/177023_5ce504536bef4395813a9ef6badf2716.html
#https://twitter.com/hadleywickham/status/664198552067829760

'''
Twitter = 280 Characters = 560 Bytes
real_seg_size = [30, 312, 460, 1531, 2950 ]  #(KB)  #Frame_size
seg_size = [35, 300, 420, 1350, 2560 ]  #(KB)  #Frame_size
video_size = [250, 1750, 3500, 7500, 25600]  #(KB)

Time_commu_Lenovo_AWSVirg = [113, 115, 120, 140, 150] #sec
Time_commu_Lenovo_EGS = [74, 76, 76, 78, 79] #sec
Time_commu_Lenovo_RPi4 = [65, 66, 66, 74, 80] #sec
Time_commu_Lenovo_NJN  = [79, 80, 82, 82, 86] #sec

thrput_commu_Lenovo_AWSVirg = [0.9*8000000 , 3*8000000 , 4*8000000 , 10*8000000 , 15*8000000] #(MB/s) -> (b/s)
thrput_commu_Lenovo_EGS = [10*8000000 , 33*8000000 , 38*8000000 , 53*8000000 , 59*8000000] #(MB/s) -> (b/s)
thrput_commu_Lenovo_RPi4 = [9*8000000 , 29*8000000 , 31*8000000 , 35*8000000 , 38*8000000] #(MB/s) -> (b/s)
thrput_commu_Lenovo_NJN = [6*8000000 , 20*8000000 , 23*8000000 , 42*8000000 , 49*8000000] #(MB/s) -> (b/s)
'''
#thrput_commu_Lenovo_EGS = [25*8000000 , 75*8000000 , 85*8000000 , 95*8000000 , 99*8000000] #(MB/s) -> (b/s)

SIZE = 208

#      	AWSVirg	       Google	  Exo	     Exo	     EGS      lenovo       NvJ       RPi4       RPi3
BW_r = [100000000 , 870000000, 840000000, 840000000, 920000000, 920000000, 450000000 , 800000000 , 328000000]#bps

# One paper had considered #VMs : 60 Cloud - 20 Cloudlet - 10 Edge


#0.015 - 0.8 ms
#65 - 85 ms
#Cloud, Tier2, Tier1(Vienna), Barcelona, Amsterdam, Paris, Brussels, Frankfurt, Graz, Ljubljana, London, Stockholm, Vienna 
#		0  1  2  3  4  5  6 7  8  9 10 11 12


# To be comparable with bw and cpu ratios.
#lambda_in = [40,40,40,40,40]
#lambda_out = [40,40,40,40,40]

T =  [[[[0]  for k in range(len(resources))] for i in range(len(tasks0))] for j in range(num_of_apps)] 
Tm = [[[[0]  for k in range(len(resources))] for i in range(len(tasks0))] for j in range(num_of_apps)] 
Tr = [[[[0]  for k in range(len(resources))] for i in range(len(tasks0))] for j in range(num_of_apps)] 
Tq = [[[[[0] for j in range(len(resources))] for k in range(10)] for i in range(len(tasks0))] for l in range(num_of_apps)] # Queuing of Data cells.


for i in range(1):
	Tm[i],Tr[i],Tq[i],T[i] = comp_times()

dictlistResources = list( {} for i in range(len(resources)) )
sorted_dictlistResources = list( {} for i in range(len(resources)) )
dictlisttasks = list( {} for i in range(len(tasks0)) )
sorted_dictlisttasks = list( {} for i in range(len(tasks0)) )

#Earliest start time
#EST = [[[[0] for j in range(len(resources))] for k in range(10)] for i in range(len(tasks0))]

'''
Tm = [[0] * len(resources) for i in range(len(tasks0))]
Tr = [[0] * len(resources) for i in range(len(tasks0))]
#print (type(Tr))
Tq = [[[[0] for j in range(len(resources))] for k in range(10)] for i in range(len(tasks0))] # Queuing of Data cells.
#print (type(Tq))
#print (len(Tq), len(Tq[0]))

Tm[0][0] = encode_20000[0] #encode_20000[][0] data size: 8sec video.
Tr[0][0] = ((video_size[index_of_segment])/BW_r[0]) + (lat[0])# + (lat[0])
Tq[0][0][0] = 0
#T[0][0] = numpy.round(numpy.round(Tm[0][0],4) + numpy.round(Tq[0][0][0],4) + numpy.round(Tr[0][0],4),4)
#print (T[0][0])

Tm[0][1] = encode_20000[1] #encode_20000[][1]
Tr[0][1] = ((video_size[index_of_segment])/BW_r[1]) + (lat[1])# + (lat[1])
Tq[0][0][1] = 0
#T[0][1] = numpy.round(numpy.round(Tm[0][1],4) + numpy.round(Tq[0][0][1],4) + numpy.round(Tr[0][1],4),4)
#print (T[0][1])

Tm[0][2] = encode_20000[2] #encode_20000[][2]
Tr[0][2] = ((video_size[index_of_segment])/BW_r[2]) + (lat[2])# + (lat[2])
Tq[0][0][2] = 0
#T[0][2] = numpy.round(numpy.round(Tm[0][2],4) + numpy.round(Tq[0][0][2],4) + numpy.round(Tr[0][2],4),4)
#print (T[0][2])

Tm[0][3] = encode_20000[3] #encode_20000[][3]
Tr[0][3] = ((video_size[index_of_segment])/BW_r[3]) + (lat[3])# + (lat[3])
Tq[0][0][3] = 0
#T[0][3] = numpy.round(numpy.round(Tm[0][3],4) + numpy.round(Tq[0][0][3],4) + numpy.round(Tr[0][3],4),4)
#print (T[0][3])

Tm[0][4] = encode_20000[4] #encode_20000[][3]
Tr[0][4] = ((video_size[index_of_segment])/BW_r[4]) + (lat[4])# + (lat[4])
Tq[0][0][4] = 0
#T[0][4] = numpy.round(numpy.round(Tm[0][4],4) + numpy.round(Tq[0][0][4],4) + numpy.round(Tr[0][4],4),4)
#print (T[0][5])

Tm[0][5] = encode_20000[5] #encode_20000[][3]
Tr[0][5] = ((video_size[index_of_segment])/BW_r[5]) + (lat[5])# + (lat[5])
Tq[0][0][5] = 0
#T[0][5] = numpy.round(numpy.round(Tm[0][5],4) + numpy.round(Tq[0][0][5],4) + numpy.round(Tr[0][5],4),4)
#print (T[0][5])

Tm[0][6] = encode_20000[6] #encode_20000[][3]
Tr[0][6] = ((video_size[index_of_segment])/BW_r[6]) + (lat[6])# + (lat[6])
Tq[0][0][6] = 0
#T[0][6] = numpy.round(numpy.round(Tm[0][6],4) + numpy.round(Tq[0][0][6],4) + numpy.round(Tr[0][6],4),4)
#print (T[0][6])

Tm[0][7] = encode_20000[7] #encode_20000[][3]
Tr[0][7] = ((video_size[index_of_segment])/BW_r[7]) + (lat[7])# + (lat[7])
Tq[0][0][7] = 0
#T[0][7] = numpy.round(numpy.round(Tm[0][7],4) + numpy.round(Tq[0][0][7],4) + numpy.round(Tr[0][7],4),4)
#print (T[0][7])

Tm[0][8] = encode_20000[8] #encode_20000[][3]
Tr[0][8] = ((video_size[index_of_segment])/BW_r[8]) + (lat[8])# + (lat[8])
Tq[0][0][8] = 0
#T[0][8] = numpy.round(numpy.round(Tm[0][8],4) + numpy.round(Tq[0][0][8],4) + numpy.round(Tr[0][8],4),4)
#print (T[0][8])

Tm[1][0] = frame_20000[0] #
#Tr[1][0] = (60*(seg_size[index_of_segment])/BW_r[0]) + (lat[0]) + (lat[0])
Tq[1][1][0] = numpy.round(Tm[1][0],4)#Tm[1][0][0]
#T[1][0] = numpy.round(numpy.round(Tm[1][0],4) + numpy.round(Tr[1][0],4),4)#+ numpy.round(Tq[1][1][0],4) 
#print (T[1][0])

Tm[1][1] = frame_20000[1] #
#Tr[1][1] = (60*(seg_size[index_of_segment])/BW_r[1]) + (lat[1]) + (lat[1])
Tq[1][1][1] = numpy.round(Tm[1][1],4)#Tm[1][0][1]
#T[1][1] = numpy.round(numpy.round(Tm[1][1],4)  + numpy.round(Tr[1][1],4),4)#+ numpy.round(Tq[1][1][1],4)
#print (T[1][1])

Tm[1][2] = frame_20000[2] #
#Tr[1][2] = (60*(seg_size[index_of_segment])/BW_r[2]) + (lat[2]) + (lat[2])
Tq[1][1][2] = numpy.round(Tm[1][2],4)#Tm[1][0][2]
#T[1][2] = numpy.round(numpy.round(Tm[1][2],4)  + numpy.round(Tr[1][2],4),4)#+ numpy.round(Tq[1][1][2],4)
#print (T[1][2])

Tm[1][3] = frame_20000[3] #
#Tr[1][3] = (60*(seg_size[index_of_segment])/BW_r[3]) + (lat[3]) + (lat[3])
Tq[1][1][3] = numpy.round(Tm[1][3],4)#Tm[1][0][3]
#T[1][3] = numpy.round(numpy.round(Tm[1][3],4) + numpy.round(Tr[1][3],4),4)#+ numpy.round(Tq[1][1][3],4) 
#print (T[1][3])

Tm[1][4] = frame_20000[4] #encode_20000[][3]
#Tr[1][4] = ((video_size[4])/BW_r[4]) + (lat[4]) + (lat[4])
Tq[1][0][4] = 0
#T[1][4] = numpy.round(numpy.round(Tm[1][4],4)  + numpy.round(Tr[1][4],4),4)#+ numpy.round(Tq[1][1][4],4)
#print (T[1][4])

Tm[1][5] = frame_20000[5] #encode_20000[][3]
#Tr[1][5] = ((video_size[4])/BW_r[5]) + (lat[5]) + (lat[5])
Tq[1][0][5] = 0
#T[1][5] = numpy.round(numpy.round(Tm[1][5],4) + numpy.round(Tr[1][5],4),4)#+ numpy.round(Tq[1][1][5],4) 
#print (T[1][5])

Tm[1][6] = frame_20000[6] #encode_20000[][3]
#Tr[1][6] = ((video_size[4])/BW_r[6]) + (lat[6]) + (lat[6])
Tq[1][0][6] = 0
#T[1][6] = numpy.round(numpy.round(Tm[1][6],4) + numpy.round(Tr[1][6],4),4)#+ numpy.round(Tq[1][1][6],4) 
#print (T[1][7])

Tm[1][7] = frame_20000[7] #encode_20000[][3]
#Tr[1][7] = ((video_size[4])/BW_r[7]) + (lat[7]) + (lat[7])
Tq[1][0][7] = 0
#T[1][7] = numpy.round(numpy.round(Tm[1][7],4) + numpy.round(Tr[1][7],4),4)#+ numpy.round(Tq[1][1][7],4) 
#print (T[1][7])

Tm[1][8] = frame_20000[8] #encode_20000[][3]
#Tr[1][8] = ((video_size[4])/BW_r[8]) + (lat[8]) + (lat[8])
Tq[1][0][8] = 0
#T[1][8] = numpy.round(numpy.round(Tm[1][8],4) + numpy.round(Tr[1][8],4),4)#+ numpy.round(Tq[1][1][8],4)
#print (T[1][8])


Tm[2][0] = Inference_high_accuracy_model[0] #encode_20000[][0] 
Tr[2][0] = (60*(seg_size[index_of_segment])/BW_r[0]) + (lat[0])# + (lat[0])
Tq[2][1][0] = numpy.round(Tm[2][0],4)#Tm[2][0][0]
#T[2][0] = numpy.round(numpy.round(Tm[2][0],4) + numpy.round(Tq[2][1][1],4) + numpy.round(Tr[2][0],4),4)
#print (T[2][0])

Tm[2][1] = Inference_high_accuracy_model[1] #encode_20000[][1]
Tr[2][1] = (60*(seg_size[index_of_segment])/BW_r[1]) + (lat[1])# + (lat[1])
Tq[2][1][1] = numpy.round(Tm[2][1],4)#Tm[2][0][1]
#T[2][1] = numpy.round(numpy.round(Tm[2][1],4) + numpy.round(Tq[2][1][1],4) + numpy.round(Tr[2][1],4),4)
#print (T[2][1])

Tm[2][2] = Inference_high_accuracy_model[2] #encode_20000[][2]
Tr[2][2] = (60*(seg_size[index_of_segment])/BW_r[2]) + (lat[2]) #+ (lat[2])
Tq[2][1][2] = numpy.round(Tm[2][2],4)#Tm[2][0][2]
#T[2][2] = numpy.round(numpy.round(Tm[2][2],4) + numpy.round(Tq[2][1][2],4) + numpy.round(Tr[2][2],4),4)
#print (T[2][2])

Tm[2][3] = Inference_high_accuracy_model[3] #encode_20000[][3]
Tr[2][3] = (60*(seg_size[index_of_segment])/BW_r[3]) + (lat[3])# + (lat[3])
Tq[2][1][3] = numpy.round(Tm[2][3],4)#Tm[2][0][3]
#T[2][3] = numpy.round(numpy.round(Tm[2][3],4) + numpy.round(Tq[2][1][3],4) + numpy.round(Tr[2][3],4),4)
#print (T[2][3])
#
Tm[2][4] = Inference_high_accuracy_model[4] #encode_20000[][3]
Tr[2][4] = (60*(seg_size[index_of_segment])/BW_r[4]) + (lat[4])# + (lat[4])
Tq[2][1][4] = 0
#T[2][4] = numpy.round(numpy.round(Tm[2][4],4) + numpy.round(Tq[2][1][4],4) + numpy.round(Tr[2][4],4),4)
#print (T[2][4])

Tm[2][5] = Inference_high_accuracy_model[5] #encode_20000[][3]
Tr[2][5] = (60*(seg_size[index_of_segment])/BW_r[5]) + (lat[5])# + (lat[5])
Tq[2][1][5] = 0
#T[2][5] = numpy.round(numpy.round(Tm[2][5],4) + numpy.round(Tq[2][1][5],4) + numpy.round(Tr[2][5],4),4)
#print (T[2][5])

Tm[2][6] = Inference_high_accuracy_model[6] #encode_20000[][3]
Tr[2][6] = (60*(seg_size[index_of_segment])/BW_r[6]) + (lat[6])# + (lat[6])
Tq[2][1][6] = 0
#T[2][6] = numpy.round(numpy.round(Tm[2][6],4) + numpy.round(Tq[2][1][6],4) + numpy.round(Tr[2][6],4),4)
#print (T[2][6])

Tm[2][7] = Inference_high_accuracy_model[7] #encode_20000[][3]
Tr[2][7] = (60*(seg_size[index_of_segment])/BW_r[7]) + (lat[7])# + (lat[7])
Tq[2][1][7] = 0
#T[2][7] = numpy.round(numpy.round(Tm[2][7],4) + numpy.round(Tq[2][1][7],4) + numpy.round(Tr[2][7],4),4)
#print (T[2][7])

Tm[2][8] = Inference_high_accuracy_model[8] #encode_20000[][3]
Tr[2][8] = (60*(seg_size[index_of_segment])/BW_r[8]) + (lat[8])# + (lat[8])
Tq[2][1][8] = 0
#T[2][8] = numpy.round(numpy.round(Tm[2][8],4) + numpy.round(Tq[2][1][8],4) + numpy.round(Tr[2][8],4),4)
#print (T[2][8])

Tm[3][0] = Low_accuracy_training_model[0] #encode_20000[][0] 
Tr[3][0] = (60*(seg_size[index_of_segment])/BW_r[0]) + (lat[0])# + (lat[0])
Tq[3][1][0] = numpy.round(Tm[3][0],4)#Tm[3][0][0]
#T[3][0] = numpy.round(numpy.round(Tm[3][0],4) + numpy.round(Tq[3][1][1],4) + numpy.round(Tr[3][0],4),4)
#print (T[3][0])

Tm[3][1] = Low_accuracy_training_model[1] #encode_20000[][1]
Tr[3][1] = (60*(seg_size[index_of_segment])/BW_r[1]) + (lat[1])# + (lat[1])
Tq[3][1][1] = numpy.round(Tm[3][1],4)#Tm[3][0][1]
#T[3][1] = numpy.round(numpy.round(Tm[3][1],4) + numpy.round(Tq[3][1][1],4) + numpy.round(Tr[3][1],4),4)
#print (T[3][1])

Tm[3][2] = Low_accuracy_training_model[2] #encode_20000[][2]
Tr[3][2] = (60*(seg_size[index_of_segment])/BW_r[2]) + (lat[2])# + (lat[2])
Tq[3][1][2] = numpy.round(Tm[3][2],4)#Tm[3][0][2]
#T[3][2] = numpy.round(numpy.round(Tm[3][2],4) + numpy.round(Tq[3][1][2],4) + numpy.round(Tr[3][2],4),4)
#print (T[3][2])

Tm[3][3] = Low_accuracy_training_model[3] #encode_20000[][3]
Tr[3][3] = (60*(seg_size[index_of_segment])/BW_r[3]) + (lat[3])# + (lat[3])
Tq[3][1][3] = numpy.round(Tm[3][3],4)#Tm[3][0][3]
#T[3][3] = numpy.round(numpy.round(Tm[3][3],4) + numpy.round(Tq[3][1][3],4) + numpy.round(Tr[3][3],4),4)
#print (T[3][3])
#
Tm[3][4] = Low_accuracy_training_model[4] #encode_20000[][3]
Tr[3][4] = (60*(seg_size[index_of_segment])/BW_r[4]) + (lat[4])# + (lat[4])
Tq[3][1][4] = 0
#T[3][4] = numpy.round(numpy.round(Tm[3][4],4) + numpy.round(Tq[3][1][4],4) + numpy.round(Tr[3][4],4),4)
#print (T[3][4])

Tm[3][5] = Low_accuracy_training_model[5] #encode_20000[][3]
Tr[3][5] = (60*(seg_size[index_of_segment])/BW_r[5]) + (lat[5])# + (lat[5])
Tq[3][1][5] = 0
#T[3][5] = numpy.round(numpy.round(Tm[3][5],4) + numpy.round(Tq[3][1][5],4) + numpy.round(Tr[3][5],4),4)
#print (T[3][5])

Tm[3][6] = Low_accuracy_training_model[6] #encode_20000[][3]
Tr[3][6] = (60*(seg_size[index_of_segment])/BW_r[6]) + (lat[6])# + (lat[6])
Tq[3][1][6] = 0
#T[3][6] = numpy.round(numpy.round(Tm[3][6],4) + numpy.round(Tq[3][1][6],4) + numpy.round(Tr[3][6],4),4)
#print (T[3][6])

Tm[3][7] = Low_accuracy_training_model[7] #encode_20000[][3]
Tr[3][7] = (60*(seg_size[index_of_segment])/BW_r[7]) + (lat[7])# + (lat[7])
Tq[3][1][7] = 0
#T[3][7] = numpy.round(numpy.round(Tm[3][7],4) + numpy.round(Tq[3][1][7],4) + numpy.round(Tr[3][7],4),4)
#print (T[3][7])

Tm[3][8] = Low_accuracy_training_model[8] #encode_20000[][3]
Tr[3][8] = (60*(seg_size[index_of_segment])/BW_r[8]) + (lat[8])# + (lat[8])
Tq[3][1][8] = 0
#T[3][8] = numpy.round(numpy.round(Tm[3][8],4) + numpy.round(Tq[3][1][8],4) + numpy.round(Tr[3][8],4),4)
#print (T[3][8])


Tm[4][0] = High_accuracy_training_model[0] #encode_20000[][0] 
Tr[4][0] = (60*(seg_size[index_of_segment])/BW_r[0]) + (lat[0])# + (lat[0])
Tq[4][1][0] = numpy.round(Tm[4][0],4)#Tm[4][0][0]
#T[4][0] = numpy.round(numpy.round(Tm[4][0],4) + numpy.round(Tq[4][1][1],4) + numpy.round(Tr[4][0],4),4)
#print (T[4][0])

Tm[4][1] = High_accuracy_training_model[1] #encode_20000[][1]
Tr[4][1] = (60*(seg_size[index_of_segment])/BW_r[1]) + (lat[1])# + (lat[1])
Tq[4][1][1] = numpy.round(Tm[4][1],4)#Tm[4][0][1]
#T[4][1] = numpy.round(numpy.round(Tm[4][1],4) + numpy.round(Tq[4][1][1],4) + numpy.round(Tr[4][1],4),4)
#print (T[4][1])

Tm[4][2] = High_accuracy_training_model[2] #encode_20000[][2]
Tr[4][2] = (60*(seg_size[index_of_segment])/BW_r[2]) + (lat[2])# + (lat[2])
Tq[4][1][2] = numpy.round(Tm[4][2],4)#Tm[4][0][2]
#T[4][2] = numpy.round(numpy.round(Tm[4][2],4) + numpy.round(Tq[4][1][2],4) + numpy.round(Tr[4][2],4),4)
#print (T[4][2])

Tm[4][3] = High_accuracy_training_model[3] #encode_20000[][3]
Tr[4][3] = (60*(seg_size[index_of_segment])/BW_r[3]) + (lat[3])# + (lat[3])
Tq[4][1][3] = numpy.round(Tm[4][3],4)#Tm[4][0][3]
#T[4][3] = numpy.round(numpy.round(Tm[4][3],4) + numpy.round(Tq[4][1][3],4) + numpy.round(Tr[4][3],4),4)
#print (T[4][3])
#
Tm[4][4] = High_accuracy_training_model[4] #encode_20000[][3]
Tr[4][4] = (60*(seg_size[index_of_segment])/BW_r[4]) + (lat[4])# + (lat[4])
Tq[4][1][4] = 0
#T[4][4] = numpy.round(numpy.round(Tm[4][4],4) + numpy.round(Tq[4][1][4],4) + numpy.round(Tr[4][4],4),4)
#print (T[4][4])

Tm[4][5] = High_accuracy_training_model[5] #encode_20000[][3]
Tr[4][5] = (60*(seg_size[index_of_segment])/BW_r[5]) + (lat[5])# + (lat[5])
Tq[4][1][5] = 0
#T[4][5] = numpy.round(numpy.round(Tm[4][5],4) + numpy.round(Tq[4][1][5],4) + numpy.round(Tr[4][5],4),4)
#print (T[4][5])

Tm[4][6] = High_accuracy_training_model[6] #encode_20000[][3]
Tr[4][6] = (60*(seg_size[index_of_segment])/BW_r[6]) + (lat[6])# + (lat[6])
Tq[4][1][6] = 0
#T[4][6] = numpy.round(numpy.round(Tm[4][6],4) + numpy.round(Tq[4][1][6],4) + numpy.round(Tr[4][6],4),4)
#print (T[4][6])

Tm[4][7] = High_accuracy_training_model[7] #encode_20000[][3]
Tr[4][7] = (60*(seg_size[index_of_segment])/BW_r[7]) + (lat[7])# + (lat[7])
Tq[4][1][7] = 0
#T[4][7] = numpy.round(numpy.round(Tm[4][7],4) + numpy.round(Tq[4][1][7],4) + numpy.round(Tr[4][7],4),4)
#print (T[4][7])

Tm[4][8] = High_accuracy_training_model[8] #encode_20000[][3]
Tr[4][8] = (60*(seg_size[index_of_segment])/BW_r[8]) + (lat[8])# + (lat[8])
Tq[4][1][8] = 0
#T[4][8] = numpy.round(numpy.round(Tm[4][8],4) + numpy.round(Tq[4][1][8],4) + numpy.round(Tr[4][8],4),4)
#print (T[4][8])

Tm[5][0] = encode_3000[0] #encode_20000[][0] 
Tr[5][0] = (60*(seg_size[index_of_segment])/BW_r[0]) + (lat[0])# + (lat[0])
Tq[5][1][0] = numpy.round(Tm[5][0],4)#Tm[5][0][0]
#T[5][0] = numpy.round(numpy.round(Tm[5][0],4) + numpy.round(Tq[5][1][1],4) + numpy.round(Tr[5][0],4),4)
#print (T[5][0])

Tm[5][1] = encode_3000[1] #encode_20000[][1]
Tr[5][1] = (60*(seg_size[index_of_segment])/BW_r[1]) + (lat[1])# + (lat[1])
Tq[5][1][1] = numpy.round(Tm[5][1],4)#Tm[5][0][1]
#T[5][1] = numpy.round(numpy.round(Tm[5][1],4) + numpy.round(Tq[5][1][1],4) + numpy.round(Tr[5][1],4),4)
#print (T[5][1])

Tm[5][2] = encode_3000[2] #encode_20000[][2]
Tr[5][2] = (60*(seg_size[index_of_segment])/BW_r[2]) + (lat[2])# + (lat[2])
Tq[5][1][2] = numpy.round(Tm[5][2],4)#Tm[5][0][2]
#T[5][2] = numpy.round(numpy.round(Tm[5][2],4) + numpy.round(Tq[5][1][2],4) + numpy.round(Tr[5][2],4),4)
#print (T[5][2])

Tm[5][3] = encode_3000[3] #encode_20000[][3]
Tr[5][3] = (60*(seg_size[index_of_segment])/BW_r[3]) + (lat[3])# + (lat[3])
Tq[5][1][3] = numpy.round(Tm[5][3],4)#Tm[5][0][3]
#T[5][3] = numpy.round(numpy.round(Tm[5][3],4) + numpy.round(Tq[5][1][3],4) + numpy.round(Tr[5][3],4),4)
#print (T[5][3])
#
Tm[5][4] = encode_3000[4] #encode_20000[][3]
Tr[5][4] = (60*(seg_size[index_of_segment])/BW_r[4]) + (lat[4])# + (lat[4])
Tq[5][1][4] = 0
#T[5][4] = numpy.round(numpy.round(Tm[5][4],4) + numpy.round(Tq[5][1][4],4) + numpy.round(Tr[5][4],4),4)
#print (T[5][4])

Tm[5][5] = encode_3000[5] #encode_20000[][3]
Tr[5][5] = (60*(seg_size[index_of_segment])/BW_r[5]) + (lat[5])# + (lat[5])
Tq[5][1][5] = 0
#T[5][5] = numpy.round(numpy.round(Tm[5][5],4) + numpy.round(Tq[5][1][5],4) + numpy.round(Tr[5][5],4),4)
#print (T[5][5])

Tm[5][6] = encode_3000[6] #encode_20000[][3]
Tr[5][6] = (60*(seg_size[index_of_segment])/BW_r[6]) + (lat[6])# + (lat[6])
Tq[5][1][6] = 0
#T[5][6] = numpy.round(numpy.round(Tm[5][6],4) + numpy.round(Tq[5][1][6],4) + numpy.round(Tr[5][6],4),4)
#print (T[5][6])

Tm[5][7] = encode_3000[7] #encode_20000[][3]
Tr[5][7] = (60*(seg_size[index_of_segment])/BW_r[7]) + (lat[7])# + (lat[7])
Tq[5][1][7] = 0
#T[5][7] = numpy.round(numpy.round(Tm[5][7],4) + numpy.round(Tq[5][1][7],4) + numpy.round(Tr[5][7],4),4)
#print (T[5][7])

Tm[5][8] = encode_3000[8] #encode_20000[][3]
Tr[5][8] = (60*(seg_size[index_of_segment])/BW_r[8]) + (lat[8])# + (lat[8])
Tq[5][1][8] = 0
#T[5][8] = numpy.round(numpy.round(Tm[5][8],4) + numpy.round(Tq[5][1][8],4) + numpy.round(Tr[5][8],4),4)



#print()
#print()
fileObject = open("D:\\00Research\\matching\\scheduler\\paper\\TPL.yaml", 'w').close()
for i in range(len(tasks0)):
	for j in range(len(resources)):		
		dictlisttasks[i][(resources[j],tasks0[i])] = numpy.round(numpy.round(Tm[i][j],4) + numpy.round(Tq[i][1][j],4), 4)
	sorted_dictlisttasks[i]=sorted(dictlisttasks[i].items(),key = itemgetter(1))
	#print((sorted_dictlisttasks[i]))
	#print(dict(sorted_dictlisttasks[i]).keys())
	tpllll=dict(dict(sorted_dictlisttasks[i]).keys())
	#print(keyyys1)
	listofvalues = list(tpllll.keys())
	listofkeys=list(tpllll.values())
	#print(listofvalues)
	dicttttt= {listofkeys[0]:listofvalues}
	#print((listofkeys[0]))
	with open(r'D:\\00Research\\matching\\scheduler\\paper\\TPL.yaml', 'a') as file:
	    documents = yaml.dump(dicttttt, file)
	#print(dict(sorted_dictlistResources[j]))
	#print()
	####print (sorted_dictlisttasks[i])
	####print()
 
#sorted(iterable, *, key=None, reverse=False)
#[(('highAccuracy', 'vm-aws'), array([0])), (('highAccuracy', 'vm-exo'), array([0])), (('highAccuracy', 't-1'), array([0])), (('highAccuracy', 'e-0'), array([0])), (('highAccuracy', 'e-1'), array([0])), (('highAccuracy', 'e-2'), array([0]))]
#print()
#print()

capfile = open('D:\\00Research\\matching\\scheduler\\paper\\capacities-testbed.yml', 'w').close()
fileObject = open("D:\\00Research\\matching\\scheduler\\paper\\RPL.yaml",'w').close()
#print (sorted(dictlisttasks.items(),key = itemgetter(1)))
for j in range(len(resources)):
	for i in range(len(tasks0)):
		dictlistResources[j][(tasks0[i],resources[j])] = numpy.round((Tr[i][j]),4)
	sorted_dictlistResources[j]=sorted(dictlistResources[j].items(),key = itemgetter(1))
	rpllll=dict(dict(sorted_dictlistResources[j]).keys())
	#print(rpllll)
	listofvalues = list(rpllll.keys())
	listofkeys=list(rpllll.values())
	#print(listofvalues)
	dicttttt= {listofkeys[0]:listofvalues}
	#print((listofkeys[0]))
	with open(r'D:\\00Research\\matching\\scheduler\\paper\\RPL.yaml', 'a') as file:
	    documents = yaml.dump(dicttttt, file)
	####print(dict(sorted_dictlistResources[j]))
	####print()
	capacity_dict= {listofkeys[0]:1}
	with open(r'D:\\00Research\\matching\\scheduler\\paper\\capacities-testbed.yml', 'a') as capfile:
	    yaml.dump(capacity_dict, capfile)


#[(('highAccuracy', 't-1'), 0), (('analysis', 't-1'), 0), (('transcoding', 't-1'), 0), (('packaging', 't-1'), 0), (('snk', 't-1'), 0), (('src', 't-1'), 0.2236), (('framing', 't-1'), 1.3697), (('lowAccuracy', 't-1'), 1.3697)]

popen = subprocess.Popen(["python" ,"D:\\00Research\\matching\\scheduler\\paper\\matchingalgo-testbed.py"], stdout=subprocess.PIPE)
popen.wait()
output = popen.stdout.read()
print (output)
'''


T  = [[0] * len(resources) for i in range(len(tasks0))]

levels = tasks0
#print (levels)

alloc = [0 for i in range(len(levels))]

for i in range(len(levels)):
	fileObject = open("D:\\00Research\\matching\\scheduler\\paper\\TPL.yaml", 'w').close()
	dictlisttasks = list( {} for i in range(len(levels)) )
	sorted_dictlisttasks = list( {} for i in range(len(levels)) )
	k = 0
	for l in range(1):
		for j in range(len(resources)):
			#if (resources[j] == "gateway"):
			#	continue
			dictlisttasks[i+(k*len(tasks0))][(resources[j],levels[i]+str(k))] = numpy.round(numpy.round(Tm[k][i][j],4)+numpy.round(Tq[k][i][1][j],4)+numpy.round(Tr[k][i][j],4),4) #numpy.round(numpy.round(Tm[k][i][j],4) + numpy.round(Tq[k][i][1][j],4), 4)
		sorted_dictlisttasks[i]=sorted(dictlisttasks[i].items(),key = itemgetter(1), reverse=False)
		#print((sorted_dictlisttasks[i]))
		#print(dict(sorted_dictlisttasks[i]).keys())
		tpllll=dict(dict(sorted_dictlisttasks[i]).keys())
		#print(keyyys1)
		listofvalues = list(tpllll.keys())
		listofkeys=list(tpllll.values())
		#print(listofvalues)
		dicttttt= {listofkeys[0]:listofvalues}
		#print((listofkeys[0]))
		with open(r'D:\\00Research\\matching\\scheduler\\paper\\TPL.yaml', 'a') as file:
		    documents = yaml.dump(dicttttt, file)
		#print(dict(sorted_dictlistResources[j]))
		#print()
		####print (sorted_dictlisttasks[i])
		####print()
	dictlistResources = list( {} for i in range(len(resources)) )
	sorted_dictlistResources = list( {} for i in range(len(resources)) )
	capfile = open('D:\\00Research\\matching\\scheduler\\paper\\capacities-testbed.yml', 'w').close()
	fileObject = open("D:\\00Research\\matching\\scheduler\\paper\\RPL.yaml",'w').close()
	for j in range(len(resources)):
		#if (resources[j] == "gateway"):
		#	continue
		#for k in range(len(resources)):
			#if (resources[k] == "gateway"):
			#	continue
			#for i in range(len(levels)):
			dictlistResources[j][(levels[i]+str(k),resources[j])] = numpy.round(numpy.round(Tm[k][i][j],4)+numpy.round(Tq[k][i][1][j],4)+numpy.round(Tr[k][i][j],4),4)
			T[i][j] =  dictlistResources[j][(levels[i]+str(k),resources[j])]
			sorted_dictlistResources[j]=sorted(dictlistResources[j].items(),key = itemgetter(1), reverse=True)
			rpllll=dict(dict(sorted_dictlistResources[j]).keys())
			#print(rpllll)
			listofvalues = list(rpllll.keys())
			listofkeys=list(rpllll.values())
			#print(listofvalues)
			dicttttt= {listofkeys[0]:listofvalues}
			#print((listofkeys[0]))
			with open(r'D:\\00Research\\matching\\scheduler\\paper\\RPL.yaml', 'a') as file:
			    documents = yaml.dump(dicttttt, file)
			####print(dict(sorted_dictlistResources[j]))
			####print()
			capacity_dict= {listofkeys[0]:2}
			with open(r'D:\\00Research\\matching\\scheduler\\paper\\capacities-testbed.yml', 'a') as capfile:
			    yaml.dump(capacity_dict, capfile)
	 ########################-----------------------------------------------------------------------------------------------------------------------------################
	 
	popen = subprocess.Popen(["python" ,"D:\\00Research\\matching\\scheduler\\paper\\matchingalgo-testbed.py"], stdout=subprocess.PIPE)
	popen.wait()
	output = popen.stdout.read()
	#print (output)
	with open(r'D:\\00Research\\matching\\scheduler\\paper\\matching-testbed.yaml') as file_read:
		# The FullLoader parameter handles the conversion from YAML
		# scalar values to the dictionary format
		matching_list = yaml.load(file_read, Loader=yaml.FullLoader)
		#print((matching_list))
		dataa = json.loads(matching_list)
		for i in range(len(dataa)):
			#print((dataa[i]))
			#print()
			for key, values in dataa[i].items():
				#print(key)
				#print((values))
				for value in ((values)):
					for task in levels:
						#print(task, " ", value)
						if (value == (task+str(k))):
							#print(tasks0.index(task))
							alloc[levels.index(task)] = i
							#print (alloc[levels.index(task)])
print (alloc)
