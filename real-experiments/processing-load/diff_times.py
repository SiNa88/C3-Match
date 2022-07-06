import numpy
import subprocess
import os
import json
import time
import sys

def comp_times(resources,task ,rec):

    #             Exo(small) Exo(med) Exo(lg)  EGS  Lenovo  NJN   RPi4   RPi3
    encode_200   = [0.44,   0.4,   0.27,        0.17, 0.33,  1.9,  2.16,   2.5] #seconds
    encode_1500  = [0.65,   0.62,   0.5,        0.36, 0.42, 2.63,  3.19,  7.35] #seconds
    encode_3000  = [0.9,   0.89,   0.65,        0.47, 0.59, 3.48,   4.4,  8.44] #seconds
    encode_6500  = [2.7,   2.58,   1.43,         1.22, 1.59, 9.68,  11.8,  22.7] #seconds
    encode_20000 = [6.2,    6.1,    3.1,         2.7, 3.16, 20.64,   28,    60] #seconds

    #             Exo(small) Exo(med) Exo(lg) EGS  Lenovo  NJN   RPi4   RPi3
    frame_200   = [0.84,     0.8,  0.5,  0.5,  0.6,    2,    4, 11]
    frame_1500  = [2.2,       1.8,   2,  2.5,    2,  9.4,   11, 20]
    frame_3000  = [2.9,       2.5,   3,  3.7,  3.1,   14,   14, 26] 
    frame_6500  = [9,       8.7,     9,   14, 13.5,   55,   49, 88]
    frame_20000 = [21,      18.5,   18,   31,   31,  117,  112, 204]

    # 		      Exo(small) Exo(med) Exo(lg)     EGS  Lenovo  NJN   RPi4   RPi3   
    inference = [     0.29,    0.26,   0.25,    0.23,  0.28,  1.94,  1.1, 1.5]
    lowtrain =   [    32,    26.5,     25.76,      17,    18,   152,  102, 1000] #seconds
    hightrain =  [     109,     103,    70.6,      33,    57,   232,  467, 1000] #seconds
        
    #    			  BG, FRA VIE Gateway Lenovo NANO  RPI4   RPI3
    download_time = [29, 38, 27,  49,     60,    61,   120,  270]

    extract_data = [12.6, 12.6, 12.6, 11.4, 11.7, 32, 107, 160]

    train_runtime = [[38,37,37,36,41,163,165,375],
                    [39,38,38,37,56,166,176,420],
                    [41,40,40,41.6,46,174,182,415],
                    [58,57,57,58,73,229,225,681],
                    [78,77,77,75,121,334,351,990]]
    # ML accuracy:   70
    train_runtime_70 = [38,37,37,36,41,163,165,375]

    # ML accuracy:   75
    train_runtime_75 = [39,38,38,37,56,166,176,420]

    # ML accuracy:   83
    train_runtime_83 = [41,40,40,41.6,46,174,182,415]

    # ML accuracy:   85
    train_runtime_85 = [58,57,57,58,73,229,225,681]

    # ML accuracy:   86
    train_runtime_86 = [78,77,77,75,121,334,351,990]

    #seg_size = 80000 #(10KB)
    #video_size = 8000000000 #(1GB)
    index_of_segment = 4
    #seg_size0 = [286720, 2457600, 3440640, 14400000, 20971520 ] #bits
    video_size = [2000000, 14000000, 28000000, 60000000, 204800000] #bits
    
    seg_size = [3.2*1024*1024*8, 3.5*1024*1024*8, 3.8*1024*1024*8, 4.1*1024*1024*8, 4.4*1024*1024*8] #bits

    #tasks0 = sys.argv[1].strip("][").split(",")
    #tasks0 = ["encoding","framing","inference","lowAccuracy","highAccuracy","transcoding","packaging"]
    ####print(tasks0)

    #              0	       1	      2      		3      4        5        6       7		8
    #resources = ["vm-aws","vm-googl","vm-exo-lg","vm-exo-med","egs","lenovo","jetson","rpi4","rpi3"]
    #resources = sys.argv[3].strip("][").split(",")

    # Converting string to list
    #tasks0 = ["encode_20000","frame_20000","hightrain","inference","package"]#sys.argv[1].strip("][").split(",")
    #print(type(tasks0))

    #tasks1 = ["download_time","extract_data","train_runtime_75","train_runtime_86"]#sys.argv[2].strip("][").split(",")

    num_of_apps = 1
    '''newtasks = [0 for i in range(len(tasks)*num_of_apps)]
    for k in range(num_of_apps):
        for i in range(len(tasks)):
            newtasks[i+(k*len(tasks))] = tasks[i]+str(k)'''

    lat =	[[ 0.5e-3, 26e-3, 15e-3 , 22.4e-3 , 22.8e-3 , 22.8e-3 , 22.8e-3 , 22.8e-3],
            [ 26e-3,  0.5e-3 , 12.5e-3 , 18.4e-3 , 18.4e-3 , 18.4e-3 , 18.4e-3 , 18.4e-3],
            [ 15e-3 , 12.5e-3 , 0.5e-3  , 7.2e-3 , 7.2e-3 , 7.5e-3 , 7.5e-3 , 7.5e-3], 
            [ 22.4e-3, 18e-3 , 7.2e-3,  0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3], 
            [ 22.8e-3, 18.4e-3, 7.2e-3, 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3], 
            [ 22.8e-3, 18.4e-3, 7.5e-3, 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3],  
            [ 22.8e-3, 18.4e-3 , 7.5e-3, 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3], 
            [ 22.8e-3, 18.4e-3 , 7.5e-3, 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3]]

    SIZE = 208

    BW_r = 	[[ 13 , 0.76,   1.5 , 0.92 , 0.9 , 0.9 , 0.85 , 0.4],
		    [ 0.76 ,  13 , 1.6 , 0.93 , 0.85 , 0.7 , 0.77, 0.4],
		    [ 1.5 , 1.6 , 13  , 0.95 , 0.9 , 0.9 , 0.9 ,0.4], 
		    [ 0.92 , 0.93 , 0.95,  0.9  , 0.86 ,  0.93 , 0.85 ,0.4], 
		    [ 0.9 , 0.85, 0.9, 0.86, 0.9, 0.92, 0.85 ,0.4 ], 
		    [ 0.9, 0.7, 0.9, 0.93, 0.92 , 0.9, 0.88 ,0.4],  
		    [ 0.85, 0.77, 0.9, 0.85, 0.85, 0.88, 0.9, 0.4 ], 
		    [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,  0.4]]

    #              AWSVirg Google Exo(lg) Exo(med) EGS  Lenovo  NvJ   RPi4   RPi3
    Time_commu_Lenovo_aws = [46, 46, 48, 59, 78] #sec
    Time_commu_Lenovo_googl = [11, 12, 12, 16, 20] #sec
    Time_commu_Lenovo_exo_vienna_lg = [7,  7,  8, 10, 12] #sec
    Time_commu_Lenovo_exo_vienna_med = [6,  7,  8, 10, 14] #sec
    Time_commu_Lenovo_exo_bg_lg = [12,  12,  12, 16, 21] #sec
    Time_commu_Lenovo_exo_frankfurt_lg = [11,  10,  10, 14, 16] #sec
    Time_commu_Lenovo_exo_muenchen_lg = [6,  7,  8, 10, 14] #sec
    Time_commu_Lenovo_exo_zuerich_lg = [11,  12,  13, 16, 21] #sec
    Time_commu_Lenovo_exo_geneva_lg = [14,  14,  14, 17, 22] #sec
    Time_commu_Lenovo_egs = [1,  1,  2,  4,  6] #sec
    Time_commu_Lenovo_rpi4 = [1,  3,  4, 12, 27] #sec
    Time_commu_Lenovo_njn  = [2,  3,  3,  6, 24] #sec

    thrput = [[2.5*8000000 , 15*8000000 , 25*8000000 , 45*8000000 , 65*8000000], #(MB/s) -> (b/s) #vie_lg
            [2.8*8000000 , 16*8000000 , 25*8000000 , 50*8000000 , 66*8000000], #(MB/s) -> (b/s) #vie_med
            [1.2*8000000 , 10*8000000 , 14*8000000 , 30*8000000 , 40*8000000], #(MB/s) -> (b/s) #bg
            [1.5*8000000 , 12*8000000 , 17*8000000 , 35*8000000 , 50*8000000], #(MB/s) -> (b/s) #frankfurt								
            [25*8000000 , 75*8000000 , 85*8000000 , 95*8000000 , 99*8000000], #(MB/s) -> (b/s) # egs
            [20*8000000 , 40*8000000 , 42*8000000 , 45*8000000 , 46*8000000], #(MB/s) -> (b/s) # rpi4
            [12*8000000 , 48*8000000 , 50*8000000 , 55*8000000 , 65*8000000]] #(MB/s) -> (b/s) # njn    

    T  = [0 for i in range(len(resources))]
    Tm = [0 for i in range(len(resources))]
    Tr = [0 for i in range(len(resources))]
    Tq = [0 for i in range(len(resources))] # Queuing of Data cells.


    if (("encode_20000") in task or (task == "package")):
        ###print(encode_20000[0]," ",type(rec))
        Tm[0] = encode_20000[0] # data size: 8sec video.
        Tr[0] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][0]) + (lat[rec][0])
        Tq[0] = 0
        T[0]  = numpy.round(numpy.round(Tm[0],4) + numpy.round(Tq[0],4) + numpy.round(Tr[0],4),4)
        #print (T[0][0])

        Tm[1] = encode_20000[1] 
        Tr[1] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][1]) + (lat[rec][1])
        Tq[1] = 0
        T[1] = numpy.round(numpy.round(Tm[1],4) + numpy.round(Tq[1],4) + numpy.round(Tr[1],4),4)
        #print (T[1])

        Tm[2] = encode_20000[2]
        Tr[2] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][2]) + (lat[rec][2])
        Tq[2] = 0
        T[2] = numpy.round(numpy.round(Tm[2],4) + numpy.round(Tq[2],4) + numpy.round(Tr[2],4),4)
        #print (T[2])

        Tm[3] = encode_20000[3]
        Tr[3] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][3]) + (lat[rec][3])
        Tq[3] = 0
        T[3] = numpy.round(numpy.round(Tm[3],4) + numpy.round(Tq[3],4) + numpy.round(Tr[3],4),4)
        #print (T[3])

        Tm[4] = encode_20000[4] 
        Tr[4] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][4]) + (lat[rec][4])
        Tq[4] = 0
        T[4] = numpy.round(numpy.round(Tm[4],4) + numpy.round(Tq[4],4) + numpy.round(Tr[4],4),4)
        #print (T[4])

        Tm[5] = encode_20000[5]
        Tr[5] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][5]) + (lat[rec][5])
        Tq[5] = 0
        T[5] = numpy.round(numpy.round(Tm[5],4) + numpy.round(Tq[5],4) + numpy.round(Tr[5],4),4)
        #print (T[5])

        Tm[6] = encode_20000[6] 
        Tr[6] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][6]) + (lat[rec][6])
        Tq[6] = 0
        T[6] = numpy.round(numpy.round(Tm[6],4) + numpy.round(Tq[6],4) + numpy.round(Tr[6],4),4)
        #print (T[6])

        Tm[7] = encode_20000[7]
        Tr[7] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][7]) + (lat[rec][7])
        Tq[7] = 0
        T[7] = numpy.round(numpy.round(Tm[7],4) + numpy.round(Tq[7],4) + numpy.round(Tr[7],4),4)
        #print (T[7])
    elif(("frame_20000") in task):
        ###print(frame_20000[0]," ",type(rec))
        Tm[0] = frame_20000[0] # data size: 8sec video.
        Tr[0] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][0]) + (lat[rec][0])
        Tq[0] = 0
        T[0]  = numpy.round(numpy.round(Tm[0],4) + numpy.round(Tq[0],4) + numpy.round(Tr[0],4),4)
        #print (T[0][0])

        Tm[1] = frame_20000[1] 
        Tr[1] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][1]) + (lat[rec][1])
        Tq[1] = 0
        T[1] = numpy.round(numpy.round(Tm[1],4) + numpy.round(Tq[1],4) + numpy.round(Tr[1],4),4)
        #print (T[1])

        Tm[2] = frame_20000[2]
        Tr[2] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][2]) + (lat[rec][2])
        Tq[2] = 0
        T[2] = numpy.round(numpy.round(Tm[2],4) + numpy.round(Tq[2],4) + numpy.round(Tr[2],4),4)
        #print (T[2])

        Tm[3] = frame_20000[3]
        Tr[3] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][3]) + (lat[rec][3])
        Tq[3] = 0
        T[3] = numpy.round(numpy.round(Tm[3],4) + numpy.round(Tq[3],4) + numpy.round(Tr[3],4),4)
        #print (T[3])

        Tm[4] = frame_20000[4] 
        Tr[4] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][4]) + (lat[rec][4])
        Tq[4] = 0
        T[4] = numpy.round(numpy.round(Tm[4],4) + numpy.round(Tq[4],4) + numpy.round(Tr[4],4),4)
        #print (T[4])

        Tm[5] = frame_20000[5]
        Tr[5] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][5]) + (lat[rec][5])
        Tq[5] = 0
        T[5] = numpy.round(numpy.round(Tm[5],4) + numpy.round(Tq[5],4) + numpy.round(Tr[5],4),4)
        #print (T[5])

        Tm[6] = frame_20000[6] 
        Tr[6] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][6]) + (lat[rec][6])
        Tq[6] = 0
        T[6] = numpy.round(numpy.round(Tm[6],4) + numpy.round(Tq[6],4) + numpy.round(Tr[6],4),4)
        #print (T[6])

        Tm[7] = frame_20000[7]
        Tr[7] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][7]) + (lat[rec][7])
        Tq[7] = 0
        T[7] = numpy.round(numpy.round(Tm[7],4) + numpy.round(Tq[7],4) + numpy.round(Tr[7],4),4)
        #print (T[7])
    elif(("hightrain") in task):
        #print(hightrain[0]," ",type(rec))
        Tm[0] = hightrain[0] # data size: 8sec video.
        Tr[0] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][0]) + (lat[rec][0])
        Tq[0] = 0
        T[0]  = numpy.round(numpy.round(Tm[0],4) + numpy.round(Tq[0],4) + numpy.round(Tr[0],4),4)
        #print (T[0][0])

        Tm[1] = hightrain[1] 
        Tr[1] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][1]) + (lat[rec][1])
        Tq[1] = 0
        T[1] = numpy.round(numpy.round(Tm[1],4) + numpy.round(Tq[1],4) + numpy.round(Tr[1],4),4)
        #print (T[1])

        Tm[2] = hightrain[2]
        Tr[2] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][2]) + (lat[rec][2])
        Tq[2] = 0
        T[2] = numpy.round(numpy.round(Tm[2],4) + numpy.round(Tq[2],4) + numpy.round(Tr[2],4),4)
        #print (T[2])

        Tm[3] = hightrain[3]
        Tr[3] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][3]) + (lat[rec][3])
        Tq[3] = 0
        T[3] = numpy.round(numpy.round(Tm[3],4) + numpy.round(Tq[3],4) + numpy.round(Tr[3],4),4)
        #print (T[3])

        Tm[4] = hightrain[4] 
        Tr[4] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][4]) + (lat[rec][4])
        Tq[4] = 0
        T[4] = numpy.round(numpy.round(Tm[4],4) + numpy.round(Tq[4],4) + numpy.round(Tr[4],4),4)
        #print (T[4])

        Tm[5] = hightrain[5]
        Tr[5] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][5]) + (lat[rec][5])
        Tq[5] = 0
        T[5] = numpy.round(numpy.round(Tm[5],4) + numpy.round(Tq[5],4) + numpy.round(Tr[5],4),4)
        #print (T[5])

        Tm[6] = hightrain[6] 
        Tr[6] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][6]) + (lat[rec][6])
        Tq[6] = 0
        T[6] = numpy.round(numpy.round(Tm[6],4) + numpy.round(Tq[6],4) + numpy.round(Tr[6],4),4)
        #print (T[6])

        Tm[7] = hightrain[7]
        Tr[7] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][7]) + (lat[rec][7])
        Tq[7] = 0
        T[7] = numpy.round(numpy.round(Tm[7],4) + numpy.round(Tq[7],4) + numpy.round(Tr[7],4),4)
        #print (T[7])
    elif(("lowtrain") in task):
        #print(lowtrain[0]," ",type(rec))
        Tm[0] = lowtrain[0] # data size: 8sec video.
        Tr[0] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][0]) + (lat[rec][0])
        Tq[0] = 0
        T[0]  = numpy.round(numpy.round(Tm[0],4) + numpy.round(Tq[0],4) + numpy.round(Tr[0],4),4)
        #print (T[0][0])

        Tm[1] = lowtrain[1] 
        Tr[1] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][1]) + (lat[rec][1])
        Tq[1] = 0
        T[1] = numpy.round(numpy.round(Tm[1],4) + numpy.round(Tq[1],4) + numpy.round(Tr[1],4),4)
        #print (T[1])

        Tm[2] = lowtrain[2]
        Tr[2] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][2]) + (lat[rec][2])
        Tq[2] = 0
        T[2] = numpy.round(numpy.round(Tm[2],4) + numpy.round(Tq[2],4) + numpy.round(Tr[2],4),4)
        #print (T[2])

        Tm[3] = lowtrain[3]
        Tr[3] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][3]) + (lat[rec][3])
        Tq[3] = 0
        T[3] = numpy.round(numpy.round(Tm[3],4) + numpy.round(Tq[3],4) + numpy.round(Tr[3],4),4)
        #print (T[3])

        Tm[4] = lowtrain[4] 
        Tr[4] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][4]) + (lat[rec][4])
        Tq[4] = 0
        T[4] = numpy.round(numpy.round(Tm[4],4) + numpy.round(Tq[4],4) + numpy.round(Tr[4],4),4)
        #print (T[4])

        Tm[5] = lowtrain[5]
        Tr[5] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][5]) + (lat[rec][5])
        Tq[5] = 0
        T[5] = numpy.round(numpy.round(Tm[5],4) + numpy.round(Tq[5],4) + numpy.round(Tr[5],4),4)
        #print (T[5])

        Tm[6] = lowtrain[6] 
        Tr[6] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][6]) + (lat[rec][6])
        Tq[6] = 0
        T[6] = numpy.round(numpy.round(Tm[6],4) + numpy.round(Tq[6],4) + numpy.round(Tr[6],4),4)
        #print (T[6])

        Tm[7] = lowtrain[7]
        Tr[7] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][7]) + (lat[rec][7])
        Tq[7] = 0
        T[7] = numpy.round(numpy.round(Tm[7],4) + numpy.round(Tq[7],4) + numpy.round(Tr[7],4),4)
        #print (T[7])
    elif(("inference") in task):
        #print(inference[0]," ",type(rec))
        Tm[0] = inference[0] # data size: 8sec video.
        Tr[0] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][0]) + (lat[rec][0])
        Tq[0] = 0
        T[0]  = numpy.round(numpy.round(Tm[0],4) + numpy.round(Tq[0],4) + numpy.round(Tr[0],4),4)
        #print (T[0][0])

        Tm[1] = inference[1] 
        Tr[1] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][1]) + (lat[rec][1])
        Tq[1] = 0
        T[1] = numpy.round(numpy.round(Tm[1],4) + numpy.round(Tq[1],4) + numpy.round(Tr[1],4),4)
        #print (T[1])

        Tm[2] = inference[2]
        Tr[2] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][2]) + (lat[rec][2])
        Tq[2] = 0
        T[2] = numpy.round(numpy.round(Tm[2],4) + numpy.round(Tq[2],4) + numpy.round(Tr[2],4),4)
        #print (T[2])

        Tm[3] = inference[3]
        Tr[3] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][3]) + (lat[rec][3])
        Tq[3] = 0
        T[3] = numpy.round(numpy.round(Tm[3],4) + numpy.round(Tq[3],4) + numpy.round(Tr[3],4),4)
        #print (T[3])

        Tm[4] = inference[4] 
        Tr[4] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][4]) + (lat[rec][4])
        Tq[4] = 0
        T[4] = numpy.round(numpy.round(Tm[4],4) + numpy.round(Tq[4],4) + numpy.round(Tr[4],4),4)
        #print (T[4])

        Tm[5] = inference[5]
        Tr[5] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][5]) + (lat[rec][5])
        Tq[5] = 0
        T[5] = numpy.round(numpy.round(Tm[5],4) + numpy.round(Tq[5],4) + numpy.round(Tr[5],4),4)
        #print (T[5])

        Tm[6] = inference[6] 
        Tr[6] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][6]) + (lat[rec][6])
        Tq[6] = 0
        T[6] = numpy.round(numpy.round(Tm[6],4) + numpy.round(Tq[6],4) + numpy.round(Tr[6],4),4)
        #print (T[6])

        Tm[7] = inference[7]
        Tr[7] = ((0.000000001)*(video_size[index_of_segment])/BW_r[rec][7]) + (lat[rec][7])
        Tq[7] = 0
        T[7] = numpy.round(numpy.round(Tm[7],4) + numpy.round(Tq[7],4) + numpy.round(Tr[7],4),4)
        #print (T[7])
    elif (("download_time") in task):
        #print(":D")
        ###print(lat[rec][7][0]," ",type(rec))
        Tm[0] = download_time[0] # data size: 8sec video.
        Tr[0] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][0]) + (lat[rec][0])
        Tq[0] = 0
        T[0]  = numpy.round(numpy.round(Tm[0],4) + numpy.round(Tq[0],4) + numpy.round(Tr[0],4),4)
        #print (T[0][0])

        Tm[1] = download_time[1] 
        Tr[1] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][1]) + (lat[rec][1])
        Tq[1] = 0
        T[1] = numpy.round(numpy.round(Tm[1],4) + numpy.round(Tq[1],4) + numpy.round(Tr[1],4),4)
        #print (T[1])

        Tm[2] = download_time[2]
        Tr[2] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][2]) + (lat[rec][2])
        Tq[2] = 0
        T[2] = numpy.round(numpy.round(Tm[2],4) + numpy.round(Tq[2],4) + numpy.round(Tr[2],4),4)
        #print (T[2])

        Tm[3] = download_time[3]
        Tr[3] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][3]) + (lat[rec][3])
        Tq[3] = 0
        T[3] = numpy.round(numpy.round(Tm[3],4) + numpy.round(Tq[3],4) + numpy.round(Tr[3],4),4)
        #print (T[3])

        Tm[4] = download_time[4] 
        Tr[4] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][4]) + (lat[rec][4])
        Tq[4] = 0
        T[4] = numpy.round(numpy.round(Tm[4],4) + numpy.round(Tq[4],4) + numpy.round(Tr[4],4),4)
        #print (T[4])

        Tm[5] = download_time[5]
        Tr[5] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][5]) + (lat[rec][5])
        Tq[5] = 0
        T[5] = numpy.round(numpy.round(Tm[5],4) + numpy.round(Tq[5],4) + numpy.round(Tr[5],4),4)
        #print (T[5])

        Tm[6] = download_time[6] 
        Tr[6] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][6]) + (lat[rec][6])
        Tq[6] = 0
        T[6] = numpy.round(numpy.round(Tm[6],4) + numpy.round(Tq[6],4) + numpy.round(Tr[6],4),4)
        #print (T[6])

        Tm[7] = download_time[7]
        Tr[7] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][7]) + (lat[rec][7])
        Tq[7] = 0
        T[7] = numpy.round(numpy.round(Tm[7],4) + numpy.round(Tq[7],4) + numpy.round(Tr[7],4),4)
        #print (T[7])
    elif(("extract_data") in task):
        ###print(extract_data[0]," ",type(rec))
        Tm[0] = extract_data[0] # data size: 8sec video.
        Tr[0] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][0]) + (lat[rec][0])
        Tq[0] = 0
        T[0]  = numpy.round(numpy.round(Tm[0],4) + numpy.round(Tq[0],4) + numpy.round(Tr[0],4),4)
        #print (T[0][0])

        Tm[1] = extract_data[1] 
        Tr[1] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][1]) + (lat[rec][1])
        Tq[1] = 0
        T[1] = numpy.round(numpy.round(Tm[1],4) + numpy.round(Tq[1],4) + numpy.round(Tr[1],4),4)
        #print (T[1])

        Tm[2] = extract_data[2]
        Tr[2] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][2]) + (lat[rec][2])
        Tq[2] = 0
        T[2] = numpy.round(numpy.round(Tm[2],4) + numpy.round(Tq[2],4) + numpy.round(Tr[2],4),4)
        #print (T[2])

        Tm[3] = extract_data[3]
        Tr[3] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][3]) + (lat[rec][3])
        Tq[3] = 0
        T[3] = numpy.round(numpy.round(Tm[3],4) + numpy.round(Tq[3],4) + numpy.round(Tr[3],4),4)
        #print (T[3])

        Tm[4] = extract_data[4] 
        Tr[4] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][4]) + (lat[rec][4])
        Tq[4] = 0
        T[4] = numpy.round(numpy.round(Tm[4],4) + numpy.round(Tq[4],4) + numpy.round(Tr[4],4),4)
        #print (T[4])

        Tm[5] = extract_data[5]
        Tr[5] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][5]) + (lat[rec][5])
        Tq[5] = 0
        T[5] = numpy.round(numpy.round(Tm[5],4) + numpy.round(Tq[5],4) + numpy.round(Tr[5],4),4)
        #print (T[5])

        Tm[6] = extract_data[6] 
        Tr[6] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][6]) + (lat[rec][6])
        Tq[6] = 0
        T[6] = numpy.round(numpy.round(Tm[6],4) + numpy.round(Tq[6],4) + numpy.round(Tr[6],4),4)
        #print (T[6])

        Tm[7] = extract_data[7]
        Tr[7] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][7]) + (lat[rec][7])
        Tq[7] = 0
        T[7] = numpy.round(numpy.round(Tm[7],4) + numpy.round(Tq[7],4) + numpy.round(Tr[7],4),4)
        #print (T[7])
    elif(("train_runtime_70") in task):
        #print(train_runtime_70[0]," ",type(rec))
        Tm[0] = train_runtime_70[0] # data size: 8sec video.
        Tr[0] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][0]) + (lat[rec][0])
        Tq[0] = 0
        T[0]  = numpy.round(numpy.round(Tm[0],4) + numpy.round(Tq[0],4) + numpy.round(Tr[0],4),4)
        #print (T[0][0])

        Tm[1] = train_runtime_70[1] 
        Tr[1] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][1]) + (lat[rec][1])
        Tq[1] = 0
        T[1] = numpy.round(numpy.round(Tm[1],4) + numpy.round(Tq[1],4) + numpy.round(Tr[1],4),4)
        #print (T[1])

        Tm[2] = train_runtime_70[2]
        Tr[2] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][2]) + (lat[rec][2])
        Tq[2] = 0
        T[2] = numpy.round(numpy.round(Tm[2],4) + numpy.round(Tq[2],4) + numpy.round(Tr[2],4),4)
        #print (T[2])

        Tm[3] = train_runtime_70[3]
        Tr[3] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][3]) + (lat[rec][3])
        Tq[3] = 0
        T[3] = numpy.round(numpy.round(Tm[3],4) + numpy.round(Tq[3],4) + numpy.round(Tr[3],4),4)
        #print (T[3])

        Tm[4] = train_runtime_70[4] 
        Tr[4] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][4]) + (lat[rec][4])
        Tq[4] = 0
        T[4] = numpy.round(numpy.round(Tm[4],4) + numpy.round(Tq[4],4) + numpy.round(Tr[4],4),4)
        #print (T[4])

        Tm[5] = train_runtime_70[5]
        Tr[5] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][5]) + (lat[rec][5])
        Tq[5] = 0
        T[5] = numpy.round(numpy.round(Tm[5],4) + numpy.round(Tq[5],4) + numpy.round(Tr[5],4),4)
        #print (T[5])

        Tm[6] = train_runtime_70[6] 
        Tr[6] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][6]) + (lat[rec][6])
        Tq[6] = 0
        T[6] = numpy.round(numpy.round(Tm[6],4) + numpy.round(Tq[6],4) + numpy.round(Tr[6],4),4)
        #print (T[6])

        Tm[7] = train_runtime_70[7]
        Tr[7] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][7]) + (lat[rec][7])
        Tq[7] = 0
        T[7] = numpy.round(numpy.round(Tm[7],4) + numpy.round(Tq[7],4) + numpy.round(Tr[7],4),4)
        #print (T[7])
    elif(("train_runtime_75") in task):
        #print(train_runtime_75[0]," ",type(rec))
        Tm[0] = train_runtime_75[0] # data size: 8sec video.
        Tr[0] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][0]) + (lat[rec][0])
        Tq[0] = 0
        T[0]  = numpy.round(numpy.round(Tm[0],4) + numpy.round(Tq[0],4) + numpy.round(Tr[0],4),4)
        #print (T[0][0])

        Tm[1] = train_runtime_75[1] 
        Tr[1] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][1]) + (lat[rec][1])
        Tq[1] = 0
        T[1] = numpy.round(numpy.round(Tm[1],4) + numpy.round(Tq[1],4) + numpy.round(Tr[1],4),4)
        #print (T[1])

        Tm[2] = train_runtime_75[2]
        Tr[2] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][2]) + (lat[rec][2])
        Tq[2] = 0
        T[2] = numpy.round(numpy.round(Tm[2],4) + numpy.round(Tq[2],4) + numpy.round(Tr[2],4),4)
        #print (T[2])

        Tm[3] = train_runtime_75[3]
        Tr[3] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][3]) + (lat[rec][3])
        Tq[3] = 0
        T[3] = numpy.round(numpy.round(Tm[3],4) + numpy.round(Tq[3],4) + numpy.round(Tr[3],4),4)
        #print (T[3])

        Tm[4] = train_runtime_75[4] 
        Tr[4] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][4]) + (lat[rec][4])
        Tq[4] = 0
        T[4] = numpy.round(numpy.round(Tm[4],4) + numpy.round(Tq[4],4) + numpy.round(Tr[4],4),4)
        #print (T[4])

        Tm[5] = train_runtime_75[5]
        Tr[5] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][5]) + (lat[rec][5])
        Tq[5] = 0
        T[5] = numpy.round(numpy.round(Tm[5],4) + numpy.round(Tq[5],4) + numpy.round(Tr[5],4),4)
        #print (T[5])

        Tm[6] = train_runtime_75[6] 
        Tr[6] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][6]) + (lat[rec][6])
        Tq[6] = 0
        T[6] = numpy.round(numpy.round(Tm[6],4) + numpy.round(Tq[6],4) + numpy.round(Tr[6],4),4)
        #print (T[6])

        Tm[7] = train_runtime_75[7]
        Tr[7] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][7]) + (lat[rec][7])
        Tq[7] = 0
        T[7] = numpy.round(numpy.round(Tm[7],4) + numpy.round(Tq[7],4) + numpy.round(Tr[7],4),4)
        #print (T[7])
    elif(("train_runtime_83") in task):
        #print(train_runtime_83[0]," ",type(rec))
        Tm[0] = train_runtime_83[0] # data size: 8sec video.
        Tr[0] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][0]) + (lat[rec][0])
        Tq[0] = 0
        T[0]  = numpy.round(numpy.round(Tm[0],4) + numpy.round(Tq[0],4) + numpy.round(Tr[0],4),4)
        #print (T[0][0])

        Tm[1] = train_runtime_83[1] 
        Tr[1] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][1]) + (lat[rec][1])
        Tq[1] = 0
        T[1] = numpy.round(numpy.round(Tm[1],4) + numpy.round(Tq[1],4) + numpy.round(Tr[1],4),4)
        #print (T[1])

        Tm[2] = train_runtime_83[2]
        Tr[2] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][2]) + (lat[rec][2])
        Tq[2] = 0
        T[2] = numpy.round(numpy.round(Tm[2],4) + numpy.round(Tq[2],4) + numpy.round(Tr[2],4),4)
        #print (T[2])

        Tm[3] = train_runtime_83[3]
        Tr[3] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][3]) + (lat[rec][3])
        Tq[3] = 0
        T[3] = numpy.round(numpy.round(Tm[3],4) + numpy.round(Tq[3],4) + numpy.round(Tr[3],4),4)
        #print (T[3])

        Tm[4] = train_runtime_83[4] 
        Tr[4] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][4]) + (lat[rec][4])
        Tq[4] = 0
        T[4] = numpy.round(numpy.round(Tm[4],4) + numpy.round(Tq[4],4) + numpy.round(Tr[4],4),4)
        #print (T[4])

        Tm[5] = train_runtime_83[5]
        Tr[5] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][5]) + (lat[rec][5])
        Tq[5] = 0
        T[5] = numpy.round(numpy.round(Tm[5],4) + numpy.round(Tq[5],4) + numpy.round(Tr[5],4),4)
        #print (T[5])

        Tm[6] = train_runtime_83[6] 
        Tr[6] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][6]) + (lat[rec][6])
        Tq[6] = 0
        T[6] = numpy.round(numpy.round(Tm[6],4) + numpy.round(Tq[6],4) + numpy.round(Tr[6],4),4)
        #print (T[6])

        Tm[7] = train_runtime_83[7]
        Tr[7] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][7]) + (lat[rec][7])
        Tq[7] = 0
        T[7] = numpy.round(numpy.round(Tm[7],4) + numpy.round(Tq[7],4) + numpy.round(Tr[7],4),4)
        #print (T[7])
    elif(("train_runtime_85") in task):
        #print(train_runtime_85[0]," ",type(rec))
        Tm[0] = train_runtime_85[0] # data size: 8sec video.
        Tr[0] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][0]) + (lat[rec][0])
        Tq[0] = 0
        T[0]  = numpy.round(numpy.round(Tm[0],4) + numpy.round(Tq[0],4) + numpy.round(Tr[0],4),4)
        #print (T[0][0])

        Tm[1] = train_runtime_85[1] 
        Tr[1] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][1]) + (lat[rec][1])
        Tq[1] = 0
        T[1] = numpy.round(numpy.round(Tm[1],4) + numpy.round(Tq[1],4) + numpy.round(Tr[1],4),4)
        #print (T[1])

        Tm[2] = train_runtime_85[2]
        Tr[2] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][2]) + (lat[rec][2])
        Tq[2] = 0
        T[2] = numpy.round(numpy.round(Tm[2],4) + numpy.round(Tq[2],4) + numpy.round(Tr[2],4),4)
        #print (T[2])

        Tm[3] = train_runtime_85[3]
        Tr[3] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][3]) + (lat[rec][3])
        Tq[3] = 0
        T[3] = numpy.round(numpy.round(Tm[3],4) + numpy.round(Tq[3],4) + numpy.round(Tr[3],4),4)
        #print (T[3])

        Tm[4] = train_runtime_85[4] 
        Tr[4] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][4]) + (lat[rec][4])
        Tq[4] = 0
        T[4] = numpy.round(numpy.round(Tm[4],4) + numpy.round(Tq[4],4) + numpy.round(Tr[4],4),4)
        #print (T[4])

        Tm[5] = train_runtime_85[5]
        Tr[5] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][5]) + (lat[rec][5])
        Tq[5] = 0
        T[5] = numpy.round(numpy.round(Tm[5],4) + numpy.round(Tq[5],4) + numpy.round(Tr[5],4),4)
        #print (T[5])

        Tm[6] = train_runtime_85[6] 
        Tr[6] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][6]) + (lat[rec][6])
        Tq[6] = 0
        T[6] = numpy.round(numpy.round(Tm[6],4) + numpy.round(Tq[6],4) + numpy.round(Tr[6],4),4)
        #print (T[6])

        Tm[7] = train_runtime_85[7]
        Tr[7] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][7]) + (lat[rec][7])
        Tq[7] = 0
        T[7] = numpy.round(numpy.round(Tm[7],4) + numpy.round(Tq[7],4) + numpy.round(Tr[7],4),4)
        #print (T[7])
    elif(("train_runtime_86") in task):
        #print(train_runtime_86[0]," ",type(rec))
        Tm[0] = train_runtime_86[0] # data size: 8sec video.
        Tr[0] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][0]) + (lat[rec][0])
        Tq[0] = 0
        T[0]  = numpy.round(numpy.round(Tm[0],4) + numpy.round(Tq[0],4) + numpy.round(Tr[0],4),4)
        #print (T[0][0])

        Tm[1] = train_runtime_86[1] 
        Tr[1] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][1]) + (lat[rec][1])
        Tq[1] = 0
        T[1] = numpy.round(numpy.round(Tm[1],4) + numpy.round(Tq[1],4) + numpy.round(Tr[1],4),4)
        #print (T[1])

        Tm[2] = train_runtime_86[2]
        Tr[2] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][2]) + (lat[rec][2])
        Tq[2] = 0
        T[2] = numpy.round(numpy.round(Tm[2],4) + numpy.round(Tq[2],4) + numpy.round(Tr[2],4),4)
        #print (T[2])

        Tm[3] = train_runtime_86[3]
        Tr[3] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][3]) + (lat[rec][3])
        Tq[3] = 0
        T[3] = numpy.round(numpy.round(Tm[3],4) + numpy.round(Tq[3],4) + numpy.round(Tr[3],4),4)
        #print (T[3])

        Tm[4] = train_runtime_86[4] 
        Tr[4] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][4]) + (lat[rec][4])
        Tq[4] = 0
        T[4] = numpy.round(numpy.round(Tm[4],4) + numpy.round(Tq[4],4) + numpy.round(Tr[4],4),4)
        #print (T[4])

        Tm[5] = train_runtime_86[5]
        Tr[5] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][5]) + (lat[rec][5])
        Tq[5] = 0
        T[5] = numpy.round(numpy.round(Tm[5],4) + numpy.round(Tq[5],4) + numpy.round(Tr[5],4),4)
        #print (T[5])

        Tm[6] = train_runtime_86[6] 
        Tr[6] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][6]) + (lat[rec][6])
        Tq[6] = 0
        T[6] = numpy.round(numpy.round(Tm[6],4) + numpy.round(Tq[6],4) + numpy.round(Tr[6],4),4)
        #print (T[6])

        Tm[7] = train_runtime_86[7]
        Tr[7] = ((0.000000001)*(SIZE*seg_size[index_of_segment])/BW_r[rec][7]) + (lat[rec][7])
        Tq[7] = 0
        T[7] = numpy.round(numpy.round(Tm[7],4) + numpy.round(Tq[7],4) + numpy.round(Tr[7],4),4)
        #print (T[7])
    
    return (Tm,Tr,Tq,T)