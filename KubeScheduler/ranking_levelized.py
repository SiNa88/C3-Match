import numpy
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter
from scipy.stats.mstats import rankdata
import networkx as nx
import yaml
import subprocess
import json
import sys
import time

from prometheus_api_client import PrometheusConnect

#start_time = time.monotonic()

prom = PrometheusConnect()

####levels = [1,2,3,4,5,6]
####print(tasks0)

# Converting string to list
tasks0 = sys.argv[1].strip("][").split(",")
####print((tasks0))

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
Inference_high_accuracy_model = [0.330 , 0.3 , 0.290 , 0.256 , 0.225 , 0.282 , 1.94 , 1.05 , 1.5]
Low_accuracy_training_model =   [25,    24,    24,       26,    17,    18,   152,   102 , 1000] #seconds
High_accuracy_training_model =  [81,   93,    84,       114,   33,    57,   232,   467 , 1000] #seconds

resources = sys.argv[2].strip("][").split(",")
####print((resources))

#              0	       1	      2      		3      4        5        6       7		8
#############resources = ["vm-aws","vm-googl","vm-exo-lg","vm-exo-med","gateway","lenovo","jetson","rpi4","rpi3"]

#101,23,11.5,11.9,0.5,0.5,0.5,0.5
#############lat = [101e-3,23e-3,11.5e-3,11.9e-3,.5e-3,.5e-3,.5e-3,.5e-3,.5e-3] #ms

#https://aws.amazon.com/kinesis/data-firehose/pricing/?nc=sn&loc=3

#seg_size = 80000 #(10KB)
#video_size = 8000000000 #(1GB)
SIZE = 208
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


#      	AWSVirg	       Google	  Exo	     Exo	     EGS      lenovo       NvJ       RPi4       RPi3
####BW_r = [100000000 , 870000000, 840000000, 840000000, 920000000, 920000000, 450000000 , 800000000 , 328000000]#bps
####for():
####	BW_rr = 0

# One project had considered #VMs : 60 Cloud - 20 Cloudlet - 10 Edge


#0.015 - 0.8 ms
#65 - 85 ms
#Cloud, Tier2, Tier1(Vienna), Barcelona, Amsterdam, Paris, Brussels, Frankfurt, Graz, Ljubljana, London, Stockholm, Vienna
#		0  1  2  3  4  5  6 7  8  9 10 11 12


# To be comparable with bw and cpu ratios.
#lambda_in = [40,40,40,40,40]
#lambda_out = [40,40,40,40,40]


Tm = [[0] * len(resources) for i in range(len(tasks0))]
Tr = [[[[0] for j in range(len(resources))] for k in range(len(resources))] for i in range(len(tasks0))]
#Tr = [[0] * len(resources) for i in range(len(tasks0))]
Tq = [[[[0] for j in range(len(resources))] for k in range(10)] for i in range(len(tasks0))] # Queuing of Data cells.
#print (type(Tq))
#print (len(Tq), len(Tq[0]))

'''
Tm[5][0] = encode_3000[0] #encode_20000[][0]
Tr[5][0] = (60*(seg_size[index_of_segment])/BW_r[0]) + (lat[0])# + (lat[0])
Tq[5][1][0] = numpy.round(Tm[5][0],4)#Tm[5][0][0]
#T[5][0] = numpy.round(numpy.round(Tm[5][0],4) + numpy.round(Tq[5][1][1],4) + numpy.round(Tr[5][0],4),4)
#print (T[5][0])
'''

## grab CPU usage of  containers
###print("CPU usage (seconds):")
query_cpu_container_encod = prom.custom_query(query="sort_desc((container_cpu_user_seconds_total{pod=~\"0encoding-.*\"}))")
avg_cpu_encod = 0
for i in range(len(query_cpu_container_encod)):
        #if(numpy.round(float(query_cpu_container[i]['value'][1]),4) >= 5):
        #print(query_cpu_container_encod[i]['metric']['node']," ",query_cpu_container_encod[i]['metric']['pod']," ",
        #numpy.round(float(query_cpu_container_encod[i]['value'][1]),5))
        avg_cpu_encod = (avg_cpu_encod + (numpy.round(float(query_cpu_container_encod[i]['value'][1]),5))) / (i + 1)
	#print(type(numpy.round(float(query_latency[i]['value'][1]),4)))
#print(query_cpu_container_encod[0]['metric']['node']," ",query_cpu_container_encod[0]['metric']['pod']," ",(avg_cpu_encod))

for i in range(len(resources)):
    Tm[0][i] = (avg_cpu_encod)

avg_cpu_fram = 0
query_cpu_container_fram = prom.custom_query(query="sort_desc((container_cpu_user_seconds_total{pod=~\"1framing-.*\"}))")
for i in range(len(query_cpu_container_fram)):
        #if(numpy.round(float(query_cpu_container[i]['value'][1]),4) >= 5):
        #print(query_cpu_container_fram[i]['metric']['node']," ",query_cpu_container_fram[i]['metric']['pod']," ",
        #numpy.round(float(query_cpu_container_fram[i]['value'][1]),5))
        avg_cpu_fram = (avg_cpu_fram + (numpy.round(float(query_cpu_container_fram[i]['value'][1]),5))) / (i + 1)
        #print(type(numpy.round(float(query_latency[i]['value'][1]),4)))
##print(query_cpu_container_fram[0]['metric']['node']," ",query_cpu_container_fram[0]['metric']['pod']," ",(avg_cpu_fram))

for i in range(len(resources)):
   Tm[1][i] = (avg_cpu_fram)


avg_cpu_train = 0
query_cpu_container_train = prom.custom_query(query="sort_desc((container_cpu_user_seconds_total{pod=~\"3training.*\"}))")
for i in range(len(query_cpu_container_train)):
        #if(numpy.round(float(query_cpu_container[i]['value'][1]),4) >= 5):
        #print(query_cpu_container_train[i]['metric']['node']," ",query_cpu_container_train[i]['metric']['pod']," ",
        #numpy.round(float(query_cpu_container_train[i]['value'][1]),5))
        avg_cpu_train = (avg_cpu_train + (numpy.round(float(query_cpu_container_train[i]['value'][1]),5))) / (i + 1)
        #print(type(numpy.round(float(query_latency[i]['value'][1]),4)))
###print(query_cpu_container_train[0]['metric']['node']," ",query_cpu_container_train[0]['metric']['pod']," ",(avg_cpu_train))

for i in range(len(resources)):
    Tm[2][i] = (avg_cpu_train)

#print()

## Communication time between two nodes
###print("Latency (seconds):")
BW_r = [13, 110, 210, 100, 115, 115, 60, 100, 41] #MBps

latency_matrix =  [[0] * len(resources) for i in range(len(resources))]

query_latency = prom.custom_query(query="sort_desc(avg(ping_durations_s{quantile='0.99'})by(source_node_name,dest_node_name))")

for i in range(len(query_latency)):
	source_node = query_latency[i]['metric']['source_node_name']
	destination_node = query_latency[i]['metric']['dest_node_name']
	if((destination_node not in resources) or (source_node not in resources)):
		continue
	if(query_latency[i]['value'][1] != "NaN"):
	#if(numpy.round(float(query_latency[i]['value'][1]),4) >= 0.0039):
	#print(query_latency[i]['metric']," ",numpy.round(float(query_latency[i]['value'][1]),4))
	#if ((query_latency[i]['metric']['source_node_name']== query_cpu_container_encod[0]['metric']['node'])
	#and (query_latency[i]['metric']['dest_node_name']  == query_cpu_container_encod[0]['metric']['node']))
		#print((numpy.round(float(query_latency[i]['value'][1]),5)))
		latency_matrix[resources.index(source_node)][resources.index(destination_node)] = (numpy.round(float(query_latency[i]['value'][1]),5))
#print(latency_matrix)
#print()

query_bw = prom.custom_query(query='(16*1024*1024)/(sum(abs(delta(download_durations_s_sum{}[1m]))< 50) by (source_node_name,dest_node_name))')
bw_matrix = [[0] * len(resources) for i in range(len(resources))]

for i in range (len(query_bw)):
        source_node = query_bw[i]['metric']['source_node_name']
        destination_node = query_bw[i]['metric']['dest_node_name']
        if((destination_node not in resources) or (source_node not in resources)):
                continue
        if(query_bw[i]['value'][1] != "+Inf"):
                bw_matrix[resources.index(source_node)][resources.index(destination_node)] = (numpy.round(float(query_bw[i]['value'][1]),5))
        #        print(bw_matrix[resources.index(source_node)][resources.index(destination_node)])
        #print(query_bw)


#print("Amount of received MB during the last 60 minutes:")
query_net = prom.custom_query(query=
#'sort_desc(instance:node_network_receive_bytes_excluding_lo:rate1m /1024 /1024 /60)'
'sort_desc(rate(node_network_receive_bytes_total{device="eth0"}[1h])  /1024/1024/3600)'
)

query_rec_traf_encod = prom.custom_query(query="sort_desc(container_network_receive_bytes_total{pod=~\"0encoding-.*\"}/1024/1024)")
query_rec_traf_fram = prom.custom_query(query="sort_desc(container_network_receive_bytes_total{pod=~\"1framing-.*\"}/1024/1024)")
query_rec_traf_train = prom.custom_query(query="sort_desc(container_network_receive_bytes_total{pod=~\"3training.*\"}/1024/1024)")

transmit_time_encod = 0
transmit_time_fram = 0
transmit_time_train = 0

for i in range(len(query_net)):
	currentnode = ("node"+(((query_net[i]['metric']['instance'])[10:11]) if((query_net[i]['metric']['instance'])[9:10] == "0") else ((query_net[i]['metric']['instance'])[9:11])))
	for j in range(len(resources)):
		if (bw_matrix[j][resources.index(currentnode)] != 0):
			Tr[0][j][resources.index(currentnode)] = (numpy.round(float(query_rec_traf_encod[0]['value'][1]),5) / (bw_matrix[j][resources.index(currentnode)]))  + latency_matrix[j][resources.index(currentnode)]
			Tr[1][j][resources.index(currentnode)] = (numpy.round(float(query_rec_traf_fram[0]['value'][1]) ,5) / (bw_matrix[j][resources.index(currentnode)])) + latency_matrix[j][resources.index(currentnode)]
			Tr[2][j][resources.index(currentnode)] = (numpy.round(float(query_rec_traf_train[0]['value'][1]),5) / (bw_matrix[j][resources.index(currentnode)])) + latency_matrix[j][resources.index(currentnode)]
			######print (Tr[2][j][resources.index(currentnode)])
			#print((BW_r[7] - numpy.round(float(query_net[i]['value'][1]),5)))

'''
Tm[5][0] = encode_3000[0] #encode_20000[][0]
Tr[5][0] = (60*(seg_size[index_of_segment])/BW_r[0]) + (lat[0])# + (lat[0])
Tq[5][1][0] = numpy.round(Tm[5][0],4)#Tm[5][0][0]
#T[5][0] = numpy.round(numpy.round(Tm[5][0],4) + numpy.round(Tq[5][1][1],4) + numpy.round(Tr[5][0],4),4)
#print (T[5][0])
'''
T  = [[0] * len(resources) for i in range(len(tasks0))]

levels = tasks0
#print (levels)

util = [0 for j in range(len(resources))]
query_node_cpu = prom.custom_query(query="(instance:node_cpu_utilisation:rate1m)")

#print (resources)
for i in range(len(query_node_cpu)):
	currentnode = ("node"+(((query_node_cpu[i]['metric']['instance'])[10:11]) if((query_node_cpu[i]['metric']['instance'])[9:10] == "0") else ((query_node_cpu[i]['metric']['instance'])[9:11])))
	if ((currentnode.find(":")) == -1):
		util[resources.index(currentnode)] = numpy.round(float(query_node_cpu[i]['value'][1]),5)
		#print (currentnode," ",numpy.round(float(query_node_cpu[i]['value'][1]),5))
util[resources.index("gateway")] = 0.8
alloc = [0 for i in range(len(levels))]



for i in range(len(levels)):
	fileObject = open("TPL.yaml", 'w').close()
	dictlisttasks = list( {} for i in range(len(levels)) )
	sorted_dictlisttasks = list( {} for i in range(len(levels)) )
	for l in range(1):
		for j in range(len(resources)):
			#if (resources[j] == "gateway"):
			#	continue
			dictlisttasks[i][(resources[j],levels[i])] = numpy.round(numpy.round(Tm[i][j],5) + numpy.round(Tq[i][1][j],5), 5)
		sorted_dictlisttasks[i] = sorted(dictlisttasks[i].items(),key = itemgetter(1), reverse=False)
		#print((sorted_dictlisttasks[i]))
		#print(dict(sorted_dictlisttasks[i]).keys())
		tpllll = dict(dict(sorted_dictlisttasks[i]).keys())
		#print(keyyys1)
		listofvalues = list(tpllll.keys())
		listofkeys = list(tpllll.values())
		#print(listofvalues)
		dicttttt= {listofkeys[0]:listofvalues}
		#print((listofkeys[0]))
		with open(r"TPL.yaml", 'a') as file:
		    documents = yaml.dump(dicttttt, file)
		#print(dict(sorted_dictlistResources[j]))
		#print()
		####print (sorted_dictlisttasks[i])
		####print()
	dictlistResources = list( {} for i in range(len(resources)) )
	sorted_dictlistResources = list( {} for i in range(len(resources)) )
	capfile = open("capacities-testbed.yml", 'w').close()
	fileObject = open("RPL.yaml", 'w').close()
	for j in range(len(resources)):
		#if (resources[j] == "gateway"):
		#	continue
		#for k in range(len(resources)):
			#if (resources[k] == "gateway"):
			#	continue
			#for i in range(len(levels)):
			if (i != 0):
				#print (alloc[i-1])
				dictlistResources[j][(levels[i],resources[j])] = numpy.round(numpy.round(Tm[i][j],5)+numpy.round(Tq[i][1][j],5)+numpy.round(Tr[i][alloc[i-1]][j],5),5)
				T[i][j] =  dictlistResources[j][(tasks0[i],resources[j])]
				sorted_dictlistResources[j] = sorted(dictlistResources[j].items(),key = itemgetter(1), reverse=True)
				rpllll = dict(dict(sorted_dictlistResources[j]).keys())
				#print(rpllll)
				listofvalues = list(rpllll.keys())
				listofkeys = list(rpllll.values())
				#print(listofvalues)
				dicttttt = {listofkeys[0]:listofvalues}
				#print((listofkeys[0]))
				with open(r'RPL.yaml', 'a') as file:
					documents = yaml.dump(dicttttt, file)
				####print(dict(sorted_dictlistResources[j]))
				cappp = float((1.00-util[j]))
				capacity_dict = {listofkeys[0]:cappp}
				with open(r'capacities-testbed.yml', 'a') as capfile:
					yaml.dump(capacity_dict, capfile)
			else:
				dictlistResources[j][(levels[i],resources[j])] = numpy.round(numpy.round(Tm[i][j],5)+numpy.round(Tq[i][1][j],5),5)
				T[i][j] =  dictlistResources[j][(tasks0[i],resources[j])]
				sorted_dictlistResources[j] = sorted(dictlistResources[j].items(),key = itemgetter(1), reverse=True)
				rpllll = dict(dict(sorted_dictlistResources[j]).keys())
				#print(rpllll)
				listofvalues = list(rpllll.keys())
				listofkeys = list(rpllll.values())
				#print(listofvalues)
				dicttttt = {listofkeys[0]:listofvalues}
				#print((listofkeys[0]))
				with open(r'RPL.yaml', 'a') as file:
					documents = yaml.dump(dicttttt, file)
				####print(dict(sorted_dictlistResources[j]))
				cappp = float((1.00-util[j]))
				capacity_dict = {listofkeys[0]:cappp}
				with open(r'capacities-testbed.yml', 'a') as capfile:
					yaml.dump(capacity_dict, capfile)
	########################-----------------------------------------------------------------------------------------------------------------------------################
	
	popen = subprocess.Popen(["python3.9" ,"matchingalgo-testbed.py"], stdout=subprocess.PIPE)
	popen.wait()
	output = popen.stdout.read()
	#print (output)
	with open(r'matching-testbed.yaml') as file_read:
		# The FullLoader parameter handles the conversion from YAML
		# scalar values to the dictionary format
		matching_list = yaml.load(file_read, Loader=yaml.FullLoader)
		#print((matching_list))
		dataa = json.loads(matching_list)
		for dd in range(len(dataa)):
			#print((dataa[dd]))
			for key, values in dataa[dd].items():
				#print(key)
				#print((values))
				for value in ((values)):
					for task in levels:
						if (value == task):
							#print(tasks0.index(task))
							alloc[levels.index(task)] = dd
							print (alloc[levels.index(task)])
#print ("Placements for ",levels," are: ", alloc)
'''
#Earliest start time
EST = [[[[0] for j in range(len(resources))] for k in range(10)] for i in range(len(levels))]

print ("=====================================================================")
print ("Microservice\t","Resource\t","EarliestStartTime\t", "ProcessTime")
print ("=====================================================================")
for row in range(len(tasks0)):
        if (row != 0):
                EST[row][1][alloc[row]] = EST[row-1][1][alloc[row-1]] + T[row-1][alloc[row-1]]
        else:
                EST[row][1][alloc[row]] = 0
        print (tasks0[row],"\t",resources[alloc[row]],"\t",numpy.round(float(EST[row][1][alloc[row]]),2),"\t\t\t", numpy.round(float(T[row][alloc[row]]),2))
        #print ("--------------------------------------------------------------------")
print ("=====================================================================")
print ("Application completion time: {} second(s)".format(numpy.round(float(EST[len(tasks0)-1][1][alloc[len(tasks0)-1]])+float(T[len(tasks0)-1][alloc[len(tasks0)-1]]),2)))
print ("=====================================================================")
EST_aws = [0 for i in range(len(tasks0))]
for row in range(len(tasks0)):
        if (row != 0):
                EST_aws[row] = EST_aws[row-1] + T[row-1][0]
        else:
                EST_aws[row] = 0
        print (tasks0[row],"\t",resources[0],"\t",numpy.round(float(EST_aws[row]),2),"\t\t\t", numpy.round(float(T[row][0]),2))
        #print ((T[row][alloc[row]]))
        print ("--------------------------------------------------------------------")
print ("Application completion time on the Cloud is: {} second(s)".format(numpy.round(float(EST_aws[len(tasks0)-1])+float(T[len(tasks0)-1][0]),2)))
'''
