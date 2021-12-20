import numpy
import yaml
import subprocess
from subprocess import call
import json
import sys
import os
import time
from topsis import topsis

start_time = time.monotonic()

tasks0 = sys.argv[1].strip("][").split(",")

#              0	       1	      2      		3      4        5        6       7		8
#resources = ["vm-aws","vm-googl","vm-exo-lg","vm-exo-med","egs","lenovo","jetson","rpi4","rpi3"]
resources = sys.argv[2].strip("][").split(",")
# C:\Users\narmehran\AppData\Local\Programs\Python\Python39\python.exe D:\00Research\matching\scheduler\paper\kcss\kcss.py [vm-aws,vm-googl,vm-exo-lg,vm-exo-med,egs,lenovo,jetson,rpi4,rpi3]


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
Inference_high_accuracy_model = [0.330, 0.3, 0.290, 0.256, 0.225, 0.282,  1.94, 1.05, 1.5] #seconds

Low_accuracy_training_model =   [25,     24,    24,    26,    17,    18,   152,  102, 1000] #seconds
High_accuracy_training_model =  [81,     93,    84,   114,    33,    57,   232,  467, 1000] #seconds


des_mat_encode = numpy.zeros((len(resources),3))
des_mat_frame = numpy.zeros((len(resources),3))
des_mat_inference_high_accuracy_model = numpy.zeros((len(resources),3))
des_mat_low_accuracy_training = numpy.zeros((len(resources),3))
des_mat_high_accuracy_training = numpy.zeros((len(resources),3))
des_mat_transcod = numpy.zeros((len(resources),3))

des_mat_encode = [[encode_20000[0],2.5000e+01,0.0000e+00],
				[encode_20000[1],1.3000e+01,1.0000e+00],
				[encode_20000[2],1.2000e+01,0.0000e+00],
				[encode_20000[3],1.2000e+01,0.0000e+00],
				[encode_20000[4],1.1000e+01,0.0000e+00],
				[encode_20000[5],1.2000e+01,0.0000e+00],
				[encode_20000[6],1.0000e+01,0.0000e+00],
				[encode_20000[7],1.1000e+01,0.0000e+00],
				[encode_20000[8],1.2000e+01,0.0000e+00]]

a_encode = des_mat_encode #[[7, 9, 9, 8, 9, 1], [8, 7, 8, 7, 9, 1], [9, 6, 8, 9, 9, 1], [6, 7, 8, 6, 9, 1]]
w = [1/3, 1/3, 1/3] #, 1/6, 1/6]
I = [-1, -1, 1] #, 1, 1]
decision_encode = topsis(a_encode, w, I)
print (decision_encode)

des_mat_frame = [[frame_20000[0],2.5000e+01,0.0000e+00],
				[frame_20000[1],1.3000e+01,1.0000e+00],
				[frame_20000[2],1.2000e+01,0.0000e+00],
				[frame_20000[3],1.2000e+01,0.0000e+00],
				[frame_20000[4],1.1000e+01,0.0000e+00],
				[frame_20000[5],1.2000e+01,0.0000e+00],
				[frame_20000[6],1.0000e+01,0.0000e+00],
				[frame_20000[7],1.1000e+01,0.0000e+00],
				[frame_20000[8],1.2000e+01,0.0000e+00]]
a_frame = des_mat_frame #[[7, 9, 9, 8, 9, 1], [8, 7, 8, 7, 9, 1], [9, 6, 8, 9, 9, 1], [6, 7, 8, 6, 9, 1]]
w = [1/3, 1/3, 1/3] #, 1/6, 1/6]
I = [-1, -1, 1] #, 1, 1]
decision_frame = topsis(a_frame, w, I)
print (decision_frame)

des_mat_inference_high_accuracy_model = [[Inference_high_accuracy_model[0],2.5000e+01,0.0000e+00],
				[Inference_high_accuracy_model[1],1.3000e+01,0.0000e+00],
				[Inference_high_accuracy_model[2],1.2000e+01,0.0000e+00],
				[Inference_high_accuracy_model[3],1.2000e+01,1.0000e+00],
				[Inference_high_accuracy_model[4],1.1000e+01,0.0000e+00],
				[Inference_high_accuracy_model[5],1.2000e+01,0.0000e+00],
				[Inference_high_accuracy_model[6],1.0000e+01,0.0000e+00],
				[Inference_high_accuracy_model[7],1.1000e+01,0.0000e+00],
				[Inference_high_accuracy_model[8],1.2000e+01,0.0000e+00]]
a_inference_high_accuracy_model = des_mat_inference_high_accuracy_model
w = [1/3, 1/3, 1/3] #, 1/6, 1/6]
I = [-1, -1, 1] #, 1, 1]
decision_inference_high_accuracy_model = topsis(a_inference_high_accuracy_model, w, I)
print (decision_inference_high_accuracy_model)


des_mat_low_accuracy_training = [[Low_accuracy_training_model[0],2.5000e+01,0.0000e+00],
				[Low_accuracy_training_model[1],1.3000e+01,0.0000e+00],
				[Low_accuracy_training_model[2],1.2000e+01,0.0000e+00],
				[Low_accuracy_training_model[3],1.2000e+01,1.0000e+00],
				[Low_accuracy_training_model[4],1.1000e+01,0.0000e+00],
				[Low_accuracy_training_model[5],1.2000e+01,0.0000e+00],
				[Low_accuracy_training_model[6],1.0000e+01,0.0000e+00],
				[Low_accuracy_training_model[7],1.1000e+01,0.0000e+00],
				[Low_accuracy_training_model[8],1.2000e+01,0.0000e+00]]
a_low_accuracy_training = des_mat_low_accuracy_training
w = [1/3, 1/3, 1/3] #, 1/6, 1/6]
I = [-1, -1, 1] #, 1, 1]
decision_low_accuracy_training = topsis(a_low_accuracy_training, w, I)
print (decision_low_accuracy_training)


des_mat_high_accuracy_training = [[High_accuracy_training_model[0],2.5000e+01,0.0000e+00],
				[High_accuracy_training_model[1],1.3000e+01,0.0000e+00],
				[High_accuracy_training_model[2],1.2000e+01,0.0000e+00],
				[High_accuracy_training_model[3],1.2000e+01,1.0000e+00],
				[High_accuracy_training_model[4],1.1000e+01,0.0000e+00],
				[High_accuracy_training_model[5],1.2000e+01,0.0000e+00],
				[High_accuracy_training_model[6],1.0000e+01,0.0000e+00],
				[High_accuracy_training_model[7],1.1000e+01,0.0000e+00],
				[High_accuracy_training_model[8],1.2000e+01,0.0000e+00]]
a_high_accuracy_training = des_mat_high_accuracy_training
w = [1/3, 1/3, 1/3] #, 1/6, 1/6]
I = [-1, -1, 1] #, 1, 1]
decision_high_accuracy_training = topsis(a_high_accuracy_training, w, I)
print (decision_high_accuracy_training)

des_mat_transcod = [[encode_3000[0],2.5000e+01,0.0000e+00],
				[encode_3000[1],1.3000e+01,1.0000e+00],
				[encode_3000[2],1.2000e+01,0.0000e+00],
				[encode_3000[3],1.2000e+01,0.0000e+00],
				[encode_3000[4],1.1000e+01,0.0000e+00],
				[encode_3000[5],1.2000e+01,0.0000e+00],
				[encode_3000[6],1.0000e+01,0.0000e+00],
				[encode_3000[7],1.1000e+01,0.0000e+00],
				[encode_3000[8],1.2000e+01,0.0000e+00]]

a_transcod = des_mat_transcod #[[7, 9, 9, 8, 9, 1], [8, 7, 8, 7, 9, 1], [9, 6, 8, 9, 9, 1], [6, 7, 8, 6, 9, 1]]
w = [1/3, 1/3, 1/3] #, 1/6, 1/6]
I = [-1, -1, 1] #, 1, 1]
decision_transcod = topsis(a_transcod, w, I)
print (decision_transcod)


elapsed_time = numpy.round(time.monotonic() - start_time , 5)
print ("=====================================================================")
print ("Algorithm execution time: {} second(s)".format(elapsed_time))
print ("=====================================================================")


'''
[[0.0000e+00 0.0000e+00 0.0000e+00 2.5000e+01 1.0000e+00]
 [1.5508e-01 3.6834e-01 7.3300e-03 1.3000e+01 1.0000e+00]
 [9.8670e-02 1.7656e-01 3.0000e-03 1.2000e+01 0.0000e+00]
 [8.9330e-02 1.7701e-01 3.3300e-03 1.2000e+01 0.0000e+00]
 [9.6580e-02 1.7104e-01 1.6700e-03 1.1000e+01 0.0000e+00]
 [9.9670e-02 1.6791e-01 3.0000e-03 1.2000e+01 0.0000e+00]
 [8.9920e-02 1.6716e-01 2.0000e-03 1.0000e+01 0.0000e+00]
 [7.5750e-02 1.4308e-01 9.0000e-03 1.1000e+01 0.0000e+00]
 [9.6080e-02 1.6799e-01 2.3300e-03 1.2000e+01 0.0000e+00]
 [8.5300e-01 5.2700e-01 3.0000e-03 1.3000e+01 0.0000e+00]
 [6.6740e-02 1.4917e-01 2.3300e-03 1.0000e+01 0.0000e+00]
 [6.5580e-02 1.4639e-01 1.3300e-03 0.0000e+00 0.0000e+00]
 [1.0708e-01 2.0319e-01 2.3300e-03 1.8000e+01 0.0000e+00]
 [1.0175e-01 1.7473e-01 2.0000e-03 1.3000e+01 0.0000e+00]
 [6.6170e-02 1.4804e-01 2.3300e-03 9.0000e+00 1.0000e+00]
 [1.1208e-01 1.8910e-01 4.6700e-03 1.3000e+01 1.0000e+00]
 [9.4670e-02 1.6606e-01 3.6700e-03 1.0000e+01 0.0000e+00]
 [1.0600e-01 1.6790e-01 4.6700e-03 1.2000e+01 0.0000e+00]]
'''