import numpy
import subprocess
import os
import json
import time
import sys

def comp_times():
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


    thrput_commu_Lenovo_aws = [0.9*8000000 , 2.8*8000000 , 4.2*8000000 , 8.5*8000000 , 11*8000000] #(MB/s) -> (b/s)
    thrput_commu_Lenovo_googl = [1.3*8000000 , 10.1*8000000 , 13.5*8000000 , 29*8000000 , 44*8000000] #(MB/s) -> (b/s)
    thrput_commu_Lenovo_exo_lg = [2.5*8000000 , 15*8000000 , 25*8000000 , 45*8000000 , 65*8000000] #(MB/s) -> (b/s)
    thrput_commu_Lenovo_exo_med = [2.8*8000000 , 16*8000000 , 25*8000000 , 50*8000000 , 66*8000000] #(MB/s) -> (b/s)
    thrput_commu_Lenovo_egs = [25*8000000 , 75*8000000 , 85*8000000 , 95*8000000 , 99*8000000] #(MB/s) -> (b/s)
    thrput_commu_Lenovo_rpi4 = [20*8000000 , 40*8000000 , 42*8000000 , 45*8000000 , 46*8000000] #(MB/s) -> (b/s)
    thrput_commu_Lenovo_njn = [12*8000000 , 48*8000000 , 50*8000000 , 55*8000000 , 65*8000000] #(MB/s) -> (b/s)

    SIZE = 208

    #      	AWSVirg		   Google	   Exo(lg)	Exo(med)	EGS       lenovo      NvJ       RPi4       RPi3
    BW_r = [100000000 , 870000000, 840000000, 840000000, 920000000, 920000000, 450000000 , 800000000 , 328000000]#bps



    T = [[0] * len(resources) for i in range(len(tasks0))]
    Tm = [[0] * len(resources) for i in range(len(tasks0))]
    Tr = [[0] * len(resources) for i in range(len(tasks0))]
    Tq = [[[[0] for j in range(len(resources))] for k in range(10)] for i in range(len(tasks0))] # Queuing of Data cells.

    #print (type(Tm))
    #print (type(Tr))
    #print (type(Tq))
    #print (len(Tq), len(Tq[0]))

    Tm[0][0] = encode_20000[0] #[][0] data size: 8sec video.
    Tr[0][0] = ((video_size[index_of_segment])/BW_r[0]) + (lat[0])# + (lat[0])
    Tq[0][0][0] = 0
    T[0][0] = numpy.round(numpy.round(Tm[0][0],4) + numpy.round(Tq[0][0][0],4) + numpy.round(Tr[0][0],4),4)
    #print (T[0][0])

    Tm[0][1] = encode_20000[1] #[][1]
    Tr[0][1] = ((video_size[index_of_segment])/BW_r[1]) + (lat[1])# + (lat[1])
    Tq[0][0][1] = 0
    T[0][1] = numpy.round(numpy.round(Tm[0][1],4) + numpy.round(Tq[0][0][1],4) + numpy.round(Tr[0][1],4),4)
    #print (T[0][1])

    Tm[0][2] = encode_20000[2] #[][2]
    Tr[0][2] = ((video_size[index_of_segment])/BW_r[2]) + (lat[2])# + (lat[2])
    Tq[0][0][2] = 0
    T[0][2] = numpy.round(numpy.round(Tm[0][2],4) + numpy.round(Tq[0][0][2],4) + numpy.round(Tr[0][2],4),4)
    #print (T[0][2])

    Tm[0][3] = encode_20000[3] #[][3]
    Tr[0][3] = ((video_size[index_of_segment])/BW_r[3]) + (lat[3])# + (lat[3])
    Tq[0][0][3] = 0
    T[0][3] = numpy.round(numpy.round(Tm[0][3],4) + numpy.round(Tq[0][0][3],4) + numpy.round(Tr[0][3],4),4)
    #print (T[0][3])

    Tm[0][4] = encode_20000[4] #[][3]
    Tr[0][4] = ((video_size[index_of_segment])/BW_r[4]) + (lat[4])# + (lat[4])
    Tq[0][0][4] = 0
    T[0][4] = numpy.round(numpy.round(Tm[0][4],4) + numpy.round(Tq[0][0][4],4) + numpy.round(Tr[0][4],4),4)
    #print (T[0][5])

    Tm[0][5] = encode_20000[5] #[][3]
    Tr[0][5] = ((video_size[index_of_segment])/BW_r[5]) + (lat[5])# + (lat[5])
    Tq[0][0][5] = 0
    T[0][5] = numpy.round(numpy.round(Tm[0][5],4) + numpy.round(Tq[0][0][5],4) + numpy.round(Tr[0][5],4),4)
    #print (T[0][5])

    Tm[0][6] = encode_20000[6] #[][3]
    Tr[0][6] = ((video_size[index_of_segment])/BW_r[6]) + (lat[6])# + (lat[6])
    Tq[0][0][6] = 0
    T[0][6] = numpy.round(numpy.round(Tm[0][6],4) + numpy.round(Tq[0][0][6],4) + numpy.round(Tr[0][6],4),4)
    #print (T[0][6])

    Tm[0][7] = encode_20000[7] #[][3]
    Tr[0][7] = ((video_size[index_of_segment])/BW_r[7]) + (lat[7])# + (lat[7])
    Tq[0][0][7] = 0
    T[0][7] = numpy.round(numpy.round(Tm[0][7],4) + numpy.round(Tq[0][0][7],4) + numpy.round(Tr[0][7],4),4)
    #print (T[0][7])

    Tm[0][8] = encode_20000[8] #[][3]
    Tr[0][8] = ((video_size[index_of_segment])/BW_r[8]) + (lat[8])# + (lat[8])
    Tq[0][0][8] = 0
    T[0][8] = numpy.round(numpy.round(Tm[0][8],4) + numpy.round(Tq[0][0][8],4) + numpy.round(Tr[0][8],4),4)
    #print (T[0][8])

    Tm[1][0] = frame_20000[0] #
    #Tr[1][0] = (60*(seg_size[index_of_segment])/BW_r[0]) + (lat[0]) + (lat[0])
    Tq[1][1][0] = 0 #numpy.round(Tm[1][0],4)#Tm[1][0][0]
    T[1][0] = numpy.round(numpy.round(Tm[1][0],4) + numpy.round(Tr[1][0],4),4)#+ numpy.round(Tq[1][1][0],4) 
    #print (T[1][0])

    Tm[1][1] = frame_20000[1] #
    #Tr[1][1] = (60*(seg_size[index_of_segment])/BW_r[1]) + (lat[1]) + (lat[1])
    Tq[1][1][1] = 0 #numpy.round(Tm[1][1],4)#Tm[1][0][1]
    T[1][1] = numpy.round(numpy.round(Tm[1][1],4)  + numpy.round(Tr[1][1],4),4)#+ numpy.round(Tq[1][1][1],4)
    #print (T[1][1])

    Tm[1][2] = frame_20000[2] #
    #Tr[1][2] = (60*(seg_size[index_of_segment])/BW_r[2]) + (lat[2]) + (lat[2])
    Tq[1][1][2] = 0 #numpy.round(Tm[1][2],4)#Tm[1][0][2]
    T[1][2] = numpy.round(numpy.round(Tm[1][2],4)  + numpy.round(Tr[1][2],4),4)#+ numpy.round(Tq[1][1][2],4)
    #print (T[1][2])

    Tm[1][3] = frame_20000[3] #
    #Tr[1][3] = (60*(seg_size[index_of_segment])/BW_r[3]) + (lat[3]) + (lat[3])
    Tq[1][1][3] = 0 #numpy.round(Tm[1][3],4)#Tm[1][0][3]
    T[1][3] = numpy.round(numpy.round(Tm[1][3],4) + numpy.round(Tr[1][3],4),4)#+ numpy.round(Tq[1][1][3],4) 
    #print (T[1][3])

    Tm[1][4] = frame_20000[4] #[][3]
    #Tr[1][4] = ((video_size[4])/BW_r[4]) + (lat[4]) + (lat[4])
    Tq[1][0][4] = 0
    T[1][4] = numpy.round(numpy.round(Tm[1][4],4)  + numpy.round(Tr[1][4],4),4)#+ numpy.round(Tq[1][1][4],4)
    #print (T[1][4])

    Tm[1][5] = frame_20000[5] #[][3]
    #Tr[1][5] = ((video_size[4])/BW_r[5]) + (lat[5]) + (lat[5])
    Tq[1][0][5] = 0
    T[1][5] = numpy.round(numpy.round(Tm[1][5],4) + numpy.round(Tr[1][5],4),4)#+ numpy.round(Tq[1][1][5],4) 
    #print (T[1][5])

    Tm[1][6] = frame_20000[6] #[][3]
    #Tr[1][6] = ((video_size[4])/BW_r[6]) + (lat[6]) + (lat[6])
    Tq[1][0][6] = 0
    T[1][6] = numpy.round(numpy.round(Tm[1][6],4) + numpy.round(Tr[1][6],4),4)#+ numpy.round(Tq[1][1][6],4) 
    #print (T[1][7])

    Tm[1][7] = frame_20000[7] #[][3]
    #Tr[1][7] = ((video_size[4])/BW_r[7]) + (lat[7]) + (lat[7])
    Tq[1][0][7] = 0
    T[1][7] = numpy.round(numpy.round(Tm[1][7],4) + numpy.round(Tr[1][7],4),4)#+ numpy.round(Tq[1][1][7],4) 
    #print (T[1][7])

    Tm[1][8] = frame_20000[8] #[][3]
    #Tr[1][8] = ((video_size[4])/BW_r[8]) + (lat[8]) + (lat[8])
    Tq[1][0][8] = 0
    T[1][8] = numpy.round(numpy.round(Tm[1][8],4) + numpy.round(Tr[1][8],4),4)#+ numpy.round(Tq[1][1][8],4)
    #print (T[1][8])

    Tm[2][0] = High_accuracy_training_model[0] #[][0] 
    Tr[2][0] = (60*(seg_size[index_of_segment])/BW_r[0]) + (lat[0])# + (lat[0])
    Tq[2][1][0] = 0 #numpy.round(Tm[2][0],4)#Tm[2][0][0]
    T[2][0] = numpy.round(numpy.round(Tm[2][0],4) + numpy.round(Tq[2][1][0],4) + numpy.round(Tr[2][0],4),4)
    #print (T[2][0])

    Tm[2][1] = High_accuracy_training_model[1] #[][1]
    Tr[2][1] = (60*(seg_size[index_of_segment])/BW_r[1]) + (lat[1])# + (lat[1])
    Tq[2][1][1] = 0 #numpy.round(Tm[2][1],4)#Tm[2][0][1]
    T[2][1] = numpy.round(numpy.round(Tm[2][1],4) + numpy.round(Tq[2][1][1],4) + numpy.round(Tr[2][1],4),4)
    #print (T[2][1])

    Tm[2][2] = High_accuracy_training_model[2] #[][2]
    Tr[2][2] = (60*(seg_size[index_of_segment])/BW_r[2]) + (lat[2])# + (lat[2])
    Tq[2][1][2] = 0 #numpy.round(Tm[2][2],4)#Tm[2][0][2]
    T[2][2] = numpy.round(numpy.round(Tm[2][2],4) + numpy.round(Tq[2][1][2],4) + numpy.round(Tr[2][2],4),4)
    #print (T[2][2])

    Tm[2][3] = High_accuracy_training_model[3] #[][3]
    Tr[2][3] = (60*(seg_size[index_of_segment])/BW_r[3]) + (lat[3])# + (lat[3])
    Tq[2][1][3] = 0 #numpy.round(Tm[2][3],4)#Tm[2][0][3]
    T[2][3] = numpy.round(numpy.round(Tm[2][3],4) + numpy.round(Tq[2][1][3],4) + numpy.round(Tr[2][3],4),4)
    #print (T[2][3])
    #
    Tm[2][4] = High_accuracy_training_model[4] #[][3]
    Tr[2][4] = (60*(seg_size[index_of_segment])/BW_r[4]) + (lat[4])# + (lat[4])
    Tq[2][1][4] = 0
    T[2][4] = numpy.round(numpy.round(Tm[2][4],4) + numpy.round(Tq[2][1][4],4) + numpy.round(Tr[2][4],4),4)
    #print (T[2][4])

    Tm[2][5] = High_accuracy_training_model[5] #[][3]
    Tr[2][5] = (60*(seg_size[index_of_segment])/BW_r[5]) + (lat[5])# + (lat[5])
    Tq[2][1][5] = 0
    T[2][5] = numpy.round(numpy.round(Tm[2][5],4) + numpy.round(Tq[2][1][5],4) + numpy.round(Tr[2][5],4),4)
    #print (T[2][5])

    Tm[2][6] = High_accuracy_training_model[6] #[][3]
    Tr[2][6] = (60*(seg_size[index_of_segment])/BW_r[6]) + (lat[6])# + (lat[6])
    Tq[2][1][6] = 0
    T[2][6] = numpy.round(numpy.round(Tm[2][6],4) + numpy.round(Tq[2][1][6],4) + numpy.round(Tr[2][6],4),4)
    #print (T[2][6])

    Tm[2][7] = High_accuracy_training_model[7] #[][3]
    Tr[2][7] = (60*(seg_size[index_of_segment])/BW_r[7]) + (lat[7])# + (lat[7])
    Tq[2][1][7] = 0
    T[2][7] = numpy.round(numpy.round(Tm[2][7],4) + numpy.round(Tq[2][1][7],4) + numpy.round(Tr[2][7],4),4)
    #print (T[2][7])

    Tm[2][8] = High_accuracy_training_model[8] #[][3]
    Tr[2][8] = (60*(seg_size[index_of_segment])/BW_r[8]) + (lat[8])# + (lat[8])
    Tq[2][1][8] = 0
    T[2][8] = numpy.round(numpy.round(Tm[2][8],4) + numpy.round(Tq[2][1][8],4) + numpy.round(Tr[2][8],4),4)
    #print (T[2][8])

    Tm[3][0] = Inference_high_accuracy_model[0] #[][0] 
    Tr[3][0] = (60*(seg_size[index_of_segment])/BW_r[0]) + (lat[0])# + (lat[0])
    Tq[3][1][0] = 0 #numpy.round(Tm[3][0],4)#Tm[3][0][0]
    T[3][0] = numpy.round(numpy.round(Tm[3][0],4) + numpy.round(Tq[3][1][0],4) + numpy.round(Tr[3][0],4),4)
    #print (T[3][0])

    Tm[3][1] = Inference_high_accuracy_model[1] #[][1]
    Tr[3][1] = (60*(seg_size[index_of_segment])/BW_r[1]) + (lat[1])# + (lat[1])
    Tq[3][1][1] = 0 #numpy.round(Tm[3][1],4)#Tm[3][0][1]
    T[3][1] = numpy.round(numpy.round(Tm[3][1],4) + numpy.round(Tq[3][1][1],4) + numpy.round(Tr[3][1],4),4)
    #print (T[3][1])

    Tm[3][2] = Inference_high_accuracy_model[2] #[][2]
    Tr[3][2] = (60*(seg_size[index_of_segment])/BW_r[2]) + (lat[2]) #+ (lat[2])
    Tq[3][1][2] = 0 #numpy.round(Tm[3][2],4)#Tm[3][0][2]
    T[3][2] = numpy.round(numpy.round(Tm[3][2],4) + numpy.round(Tq[3][1][2],4) + numpy.round(Tr[3][2],4),4)
    #print (T[3][2])

    Tm[3][3] = Inference_high_accuracy_model[3] #[][3]
    Tr[3][3] = (60*(seg_size[index_of_segment])/BW_r[3]) + (lat[3])# + (lat[3])
    Tq[3][1][3] = 0 #numpy.round(Tm[3][3],4)#Tm[3][0][3]
    T[3][3] = numpy.round(numpy.round(Tm[3][3],4) + numpy.round(Tq[3][1][3],4) + numpy.round(Tr[3][3],4),4)
    #print (T[3][3])
    #
    Tm[3][4] = Inference_high_accuracy_model[4] #[][3]
    Tr[3][4] = (60*(seg_size[index_of_segment])/BW_r[4]) + (lat[4])# + (lat[4])
    Tq[3][1][4] = 0
    T[3][4] = numpy.round(numpy.round(Tm[3][4],4) + numpy.round(Tq[3][1][4],4) + numpy.round(Tr[3][4],4),4)
    #print (T[3][4])

    Tm[3][5] = Inference_high_accuracy_model[5] #[][3]
    Tr[3][5] = (60*(seg_size[index_of_segment])/BW_r[5]) + (lat[5])# + (lat[5])
    Tq[3][1][5] = 0
    T[3][5] = numpy.round(numpy.round(Tm[3][5],4) + numpy.round(Tq[3][1][5],4) + numpy.round(Tr[3][5],4),4)
    #print (T[3][5])

    Tm[3][6] = Inference_high_accuracy_model[6] #[][3]
    Tr[3][6] = (60*(seg_size[index_of_segment])/BW_r[6]) + (lat[6])# + (lat[6])
    Tq[3][1][6] = 0
    T[3][6] = numpy.round(numpy.round(Tm[3][6],4) + numpy.round(Tq[3][1][6],4) + numpy.round(Tr[3][6],4),4)
    #print (T[3][6])

    Tm[3][7] = Inference_high_accuracy_model[7] #[][3]
    Tr[3][7] = (60*(seg_size[index_of_segment])/BW_r[7]) + (lat[7])# + (lat[7])
    Tq[3][1][7] = 0
    T[3][7] = numpy.round(numpy.round(Tm[3][7],4) + numpy.round(Tq[3][1][7],4) + numpy.round(Tr[3][7],4),4)
    #print (T[3][7])

    Tm[3][8] = Inference_high_accuracy_model[8] #[][3]
    Tr[3][8] = (60*(seg_size[index_of_segment])/BW_r[8]) + (lat[8])# + (lat[8])
    Tq[3][1][8] = 0
    T[3][8] = numpy.round(numpy.round(Tm[3][8],4) + numpy.round(Tq[3][1][8],4) + numpy.round(Tr[3][8],4),4)
    #print (T[3][8])

    Tm[4][0] = encode_20000[0] #[][0] 
    Tr[4][0] = (60*(seg_size[index_of_segment])/BW_r[0]) + (lat[0])# + (lat[0])
    Tq[4][1][0] = 0 #numpy.round(Tm[4][0],4)#Tm[4][0][0]
    T[4][0] = numpy.round(numpy.round(Tm[4][0],4) + numpy.round(Tq[4][1][0],4) + numpy.round(Tr[4][0],4),4)
    #print (T[4][0])

    Tm[4][1] = encode_20000[1] #[][1]
    Tr[4][1] = (60*(seg_size[index_of_segment])/BW_r[1]) + (lat[1])# + (lat[1])
    Tq[4][1][1] = 0 #numpy.round(Tm[4][1],4)#Tm[4][0][1]
    T[4][1] = numpy.round(numpy.round(Tm[4][1],4) + numpy.round(Tq[4][1][1],4) + numpy.round(Tr[4][1],4),4)
    #print (T[4][1])

    Tm[4][2] = encode_20000[2] #[][2]
    Tr[4][2] = (60*(seg_size[index_of_segment])/BW_r[2]) + (lat[2])# + (lat[2])
    Tq[4][1][2] = 0 #numpy.round(Tm[4][2],4)#Tm[4][0][2]
    T[4][2] = numpy.round(numpy.round(Tm[4][2],4) + numpy.round(Tq[4][1][2],4) + numpy.round(Tr[4][2],4),4)
    #print (T[4][2])

    Tm[4][3] = encode_20000[3] #[][3]
    Tr[4][3] = (60*(seg_size[index_of_segment])/BW_r[3]) + (lat[3])# + (lat[3])
    Tq[4][1][3] = 0 #numpy.round(Tm[4][3],4)#Tm[4][0][3]
    T[4][3] = numpy.round(numpy.round(Tm[4][3],4) + numpy.round(Tq[4][1][3],4) + numpy.round(Tr[4][3],4),4)
    #print (T[4][3])
    #
    Tm[4][4] = encode_20000[4] #[][3]
    Tr[4][4] = (60*(seg_size[index_of_segment])/BW_r[4]) + (lat[4])# + (lat[4])
    Tq[4][1][4] = 0
    T[4][4] = numpy.round(numpy.round(Tm[4][4],4) + numpy.round(Tq[4][1][4],4) + numpy.round(Tr[4][4],4),4)
    #print (T[4][4])

    Tm[4][5] = encode_20000[5] #[][3]
    Tr[4][5] = (60*(seg_size[index_of_segment])/BW_r[5]) + (lat[5])# + (lat[5])
    Tq[4][1][5] = 0
    T[4][5] = numpy.round(numpy.round(Tm[4][5],4) + numpy.round(Tq[4][1][5],4) + numpy.round(Tr[4][5],4),4)
    #print (T[4][5])

    Tm[4][6] = encode_20000[6] #[][3]
    Tr[4][6] = (60*(seg_size[index_of_segment])/BW_r[6]) + (lat[6])# + (lat[6])
    Tq[4][1][6] = 0
    T[4][6] = numpy.round(numpy.round(Tm[4][6],4) + numpy.round(Tq[4][1][6],4) + numpy.round(Tr[4][6],4),4)
    #print (T[4][6])

    Tm[4][7] = encode_20000[7] #[][3]
    Tr[4][7] = (60*(seg_size[index_of_segment])/BW_r[7]) + (lat[7])# + (lat[7])
    Tq[4][1][7] = 0
    T[4][7] = numpy.round(numpy.round(Tm[4][7],4) + numpy.round(Tq[4][1][7],4) + numpy.round(Tr[4][7],4),4)
    #print (T[4][7])

    Tm[4][8] = encode_20000[8] #[][3]
    Tr[4][8] = (60*(seg_size[index_of_segment])/BW_r[8]) + (lat[8])# + (lat[8])
    Tq[4][1][8] = 0
    T[4][8] = numpy.round(numpy.round(Tm[4][8],4) + numpy.round(Tq[4][1][8],4) + numpy.round(Tr[4][8],4),4)


    return (Tm,Tr,Tq,T)