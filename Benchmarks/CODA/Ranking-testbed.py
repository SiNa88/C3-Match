import numpy
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter
from scipy.stats.mstats import rankdata    
import yaml
import subprocess
import json
import sys

sys.path.append('./diff_times')
from diff_times import comp_times

tasks0 = sys.argv[1].strip("][").split(",")

#              0	       1	      2      		3      4        5        6       7		8
#resources = ["vm-aws","vm-googl","vm-exo-lg","vm-exo-med","egs","lenovo","jetson","rpi4","rpi3"]
resources = sys.argv[2].strip("][").split(",")
#101,23,11.5,11.9,0.5,0.5,0.5,0.5
lat = [101e-3,23e-3,11.5e-3,11.9e-3,.5e-3,.5e-3,.5e-3,.5e-3,.5e-3] #ms
#print (len(lat))

#             EGS   Lenovo NvJ   RPi4  RPi3
encode_200  = [0.17,0.33,1.9,2.16, 2.5]
encode_1500 = [0.36,0.42,2.63,3.19,7.35]
encode_3000 = [0.47,0.59,3.48,4.4,8.44]
encode_6500 = [1.22,1.59,9.68,11.8,22.7]
#encode_12000 = [2.39,3.07,20,23.62,48.6]
encode_20000 = [2.69,3.16,20.64,28,60]

#encode_HD  = [0.4 , 0.4 , 3.6 , 4.8 , 30.6]
#encode_FHD = [0.5 , 0.5 , 4 , 5.5 , 32.3]
#encode_QHD = [0.9 , 0.9 , 6.4 , 8.2 , 36.3]

#                     EGS Lenovo NvJ  RPi4 RPi3   
Low_accuracy_training_model = [16.99, 17.8 , 151.4  , 101.864 , 1000]
High_accuracy_training_model = [33.23, 56.746 , 232.253 , 466.830 , 1000]
#CNN_training_model = [115 , 65 , 432  , 1050 , 1630]
#QNN_training_model = [54.1 , 35.6 , 40.1 , 6.2]



# From: https://link.springer.com/chapter/10.1007/978-3-030-03596-9_14
#lat = [85e-3,60e-3,25e-3,22e-3,27e-3,21e-3,13e-3,17e-3,9e-3,28e-3,43e-3,21e-3,1e-3,1e-3] #ms

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


#https://aws.amazon.com/kinesis/data-firehose/pricing/?nc=sn&loc=3
#seg_size = 80000 #(10KB)
#video_size = 80000000000 #(10GB)


#seg_size = 80000 #(10KB)
#video_size = 8000000000 #(1GB)

seg_size = [280000, 2400000, 3360000, 10800000, 20480000 ] #bits
video_size = [2000000, 14000000, 28000000, 60000000, 204800000] #bits
'''
Twitter = 280 Characters = 560 Bytes
seg_size = [35, 300, 420, 1350, 2560 ] #(KB)  #Frame_size
video_size = [250, 1750, 3500, 7500, 25600] #(KB)

Time_commu_Lenovo_EGC = [74, 76, 76, 78, 79]
Time_commu_Lenovo_RPi4 = [65, 66, 66, 74, 80]
Time_commu_Lenovo_NJN  = [79, 80, 82, 82, 86]

bw_commu_Lenovo_EGC = [12 , 32 , 40 , 50 , 59] #(MB/s)
bw_commu_Lenovo_RPi4 = [9, 25,  30, 40, 45] #(MB/s)
bw_commu_Lenovo_NJN = [3.5, 18, 22, 40, 49] #(MB/s)
'''

#0.015 - 0.8 ms
#65 - 85 ms
#Cloud, Tier2, Tier1(Vienna), Barcelona, Amsterdam, Paris, Brussels, Frankfurt, Graz, Ljubljana, London, Stockholm, Vienna 
#		0  1  2  3  4  5  6 7  8  9 10 11 12


# To be comparable with bw and cpu ratios.
lambda_in = [1,1,1,1]
lambda_out = [1,1,1,1]

num_of_apps = 1

newtasks = [0 for i in range(len(tasks0)*num_of_apps)]

for k in range(num_of_apps):
	for i in range(len(tasks0)):
		newtasks[i+(k*len(tasks0))] = tasks0[i]+str(k)
		#print (i," ",(k))

T = [[[[0] for k in range(len(resources))] for i in range(len(tasks0))] for j in range(num_of_apps)] 
Tm = [[[[0] for k in range(len(resources))] for i in range(len(tasks0))] for j in range(num_of_apps)] 
Tr = [[[[0] for k in range(len(resources))] for i in range(len(tasks0))] for j in range(num_of_apps)] 
Tq = [[[[[0] for j in range(len(resources))] for k in range(10)] for i in range(len(tasks0))] for l in range(num_of_apps)] # Queuing of Data cells.

for i in range(num_of_apps):
	Tm[i],Tr[i],Tq[i],T[i] = comp_times()

dictlistResources = list( {} for i in range(len(resources)) )
sorted_dictlistResources = list( {} for i in range(len(resources)) )
dictlisttasks = list( {} for i in range(len(tasks0)) )
sorted_dictlisttasks = list( {} for i in range(len(tasks0)) )

'''print()
for i in range(len(tasks0)):
	for j in range(len(resources)):		
		dictlisttasks[i][(tasks0[i],resources[j])] = numpy.round((T[i][j]),4)

#sorted(iterable, *, key=None, reverse=False)
for i in range(len(tasks0)):
	sorted_dictlisttasks[i]=sorted(dictlisttasks[i].items(),key = itemgetter(1))
	print (sorted_dictlisttasks[i])
	print ()

#print (sorted(dictlisttasks0.items(),key = itemgetter(1)))
for j in range(len(resources)):
	for i in range(len(tasks0)):
		dictlistResources[j][(tasks0[i],resources[j])] = numpy.round((BW_r[j]) - (seg_size[4]),4)
	sorted_dictlistResources[j]=sorted(dictlistResources[j].items(),key = itemgetter(1))
	print (sorted_dictlistResources[j])
	print ()
'''


mytpl = dict()
#print()
#print()
fileObject = open("D:\\00Research\\matching\\scheduler\\paper\\CODA\\TPL.yaml", 'w').close()
for k in range(num_of_apps):
	for i in range(len(tasks0)):
		for j in range(len(resources)):		
			dictlisttasks[i+(k*len(tasks0))][(resources[j],tasks0[i]+str(k))] = numpy.round(numpy.round(T[k][i][j],4),4)
		sorted_dictlisttasks[i+(k*len(tasks0))]=sorted(dictlisttasks[i+(k*len(tasks0))].items(),key = itemgetter(1), reverse=False)
		#print((sorted_dictlisttasks[i]))
		#print(dict(sorted_dictlisttasks[i]).keys())
		tpllll=dict(dict(sorted_dictlisttasks[i+(k*len(tasks0))]).keys())
		#mytpl = tpllll
		#print (k," ",i)
		#print(keyyys1)
		listofvalues = list(tpllll.keys())
		listofkeys=list(tpllll.values())
		#print(listofvalues)
		dicttttt= {listofkeys[0]:listofvalues}
		#print((listofkeys[0]))
		with open(r'D:\\00Research\\matching\\scheduler\\paper\\CODA\\TPL.yaml', 'a') as file:
			documents = yaml.dump(dicttttt, file)
		#print(dict(sorted_dictlistResources[j]))
		#print()
		####print (sorted_dictlisttasks[i])
		####print()
 
#sorted(iterable, *, key=None, reverse=False)
#[(('highAccuracy', 'vm-aws'), array([0])), (('highAccuracy', 'vm-exo'), array([0])), (('highAccuracy', 't-1'), array([0])), (('highAccuracy', 'e-0'), array([0])), (('highAccuracy', 'e-1'), array([0])), (('highAccuracy', 'e-2'), array([0]))]
#print()
#print()
myrpl = dict()
capfile = open('D:\\00Research\\matching\\scheduler\\paper\\CODA\\capacities-testbed.yml', 'w').close()
fileObject = open("D:\\00Research\\matching\\scheduler\\paper\\CODA\\RPL.yaml",'w').close()
#print (sorted(dictlisttasks.items(),key = itemgetter(1)))
for j in range(len(resources)):
	for k in range(num_of_apps):
		for i in range(len(tasks0)):
			dictlistResources[j][(tasks0[i]+str(k),resources[j])] = numpy.round((BW_r[j]) - (seg_size[4]),4) #numpy.round(Tm[k][i][j],4)+numpy.round(Tq[k][i][1][j],4)+
	sorted_dictlistResources[j]=sorted(dictlistResources[j].items(),key = itemgetter(1), reverse=True)
	rpllll=dict(dict(sorted_dictlistResources[j]).keys())
	#myrpl = rpllll
	#print(rpllll)
	listofvalues = list(rpllll.keys())
	listofkeys=list(rpllll.values())
	#print(listofvalues)
	dicttttt= {listofkeys[0]:listofvalues}
	#print((listofkeys[0]))
	with open(r'D:\\00Research\\matching\\scheduler\\paper\\CODA\\RPL.yaml', 'a') as file:
		documents = yaml.dump(dicttttt, file)
	####print(dict(sorted_dictlistResources[j]))
		####print()
	capacity_dict= {listofkeys[0]:2}
	with open(r'D:\\00Research\\matching\\scheduler\\paper\\CODA\\capacities-testbed.yml', 'a') as capfile:
		yaml.dump(capacity_dict, capfile)



#[(('highAccuracy', 't-1'), 0), (('analysis', 't-1'), 0), (('transcoding', 't-1'), 0), (('packaging', 't-1'), 0), (('snk', 't-1'), 0), (('src', 't-1'), 0.2236), (('framing', 't-1'), 1.3697), (('lowAccuracy', 't-1'), 1.3697)]

alloc = [0 for i in range(len(newtasks))]

popen = subprocess.Popen(["C:\\Users\\narmehran\\AppData\\Local\\Programs\\Python\\Python39\\python.exe" ,"D:\\00Research\\matching\\scheduler\\paper\\CODA\\matchingalgo-testbed.py"], stdout=subprocess.PIPE)
popen.wait()
output = popen.stdout.read()
#print (output)
with open(r'D:\\00Research\\matching\\scheduler\\paper\\CODA\\matching-testbed.yaml') as file_read:
	# The FullLoader parameter handles the conversion from YAML
	# scalar values to the dictionary format
	matching_list = yaml.load(file_read, Loader=yaml.FullLoader)
	#print((matching_list))
	dataa = json.loads(matching_list)
	for i in range(len(dataa)):
		#print((dataa[i]))
		for key, values in dataa[i].items():
			#print(key)
			#print((values))
			for value in ((values)):
				for task in newtasks:
					if (value == task):
						#print(newtasks.index(task))
						alloc[newtasks.index(task)] = i
						#print (task, " ", alloc[newtasks.index(task)])
print ((alloc))


summatching = [0 for i in range(3)]
sumedge = [0 for i in range(3)]
sumcloud = [0 for i in range(3)]
max_summatching = 0	
max_sumedge = 0
max_sumcloud = 0	
#Application consisiting of some tasks0: (it consists of paths from source to destination)
#paths = [["encoding","lowAccuracy","transcoding","packaging"],
#		  ["encoding","lowAccuracy","highAccuracy","analysis","packaging"],
#		  ["encoding","lowAccuracy","transcoding","analysis","packaging"]]
'''
#				          0		            1	             2                 3         4              5
tasks0 = ["encoding", "lowAccuracy", "detect_classify ", "transcoding", "per_storing", "packaging"]
#	0		 1	   2     3     4     5     6     7     8     9    10    11    12    13
resources = ["vm1-cdc","t-2","t-1","e-0","e-1","e-2","e-3","e-4","e-5","e-6","e-7","e-8","e-9","e-10"]
'''

'''
#fruits.index("cherry")
if (T[0][10] != 0 and  T[1][1] != 0 and  T[3][2] != 0 ):
		summatching[0] = summatching[0] + T[0][10] + T[1][1] + T[3][2] 
		summatching[1] = summatching[1] + T[0][10] + lat[2] + T[1][1] - (2*lat[1]) + T[2][1] + T[4][2] - lat[2] - lat[1] 
		summatching[2] = summatching[2] + T[0][10] + lat[2] + T[1][1] +  T[3][2] - (2*lat[1]) 
		print("Respone in Diff. branches: " , summatching[0],' ',summatching[1],' ',summatching[2])
		max_summatching = max(summatching[0],summatching[1],summatching[2])

if(T[0][2] != 0 and T[1][2] != 0 and  T[3][2] != 0 ):
		sumedge[0]     = sumedge[0]  + T[0][2] + T[1][2] + T[3][2]
		sumedge[1]     = sumedge[1]  + T[0][2] + T[1][2]  
		sumedge[2]     = sumedge[2]  + T[0][2] + T[1][2] + T[3][2] 
		print ("Respone in Diff. branches: " , sumedge[0],' ',sumedge[1],' ',sumedge[2])
		max_sumedge = max(sumedge[0],sumedge[1],sumedge[2])

if(T[0][0] != 0 and T[1][0] != 0 and T[3][0]  != 0 ):
		sumcloud[0]    = sumcloud[0]    +  lat[2] + lat[1] + T[0][0] - (2*lat[0]) + T[1][0] - (2*lat[0])+T[3][0]- (2*lat[0])+T[5][0] + lat[2] + lat[1]
		sumcloud[1]    = sumcloud[1]    +  lat[2] + lat[1] + T[0][0] - (2*lat[0]) + T[1][0] - (2*lat[0])+T[2][0]- (2*lat[0])+T[4][0]- (2*lat[0])+T[5][0] + lat[2] + lat[1]
		sumcloud[2]    = sumcloud[2]    +  lat[2] + lat[1] + T[0][0] - (2*lat[0]) + T[1][0] - (2*lat[0])+T[3][0]- (2*lat[0])+T[4][0]- (2*lat[0])+T[5][0] + lat[2] + lat[1]
		print ("Respone in Diff. branches: " , sumcloud[0],' ',sumcloud[1],' ',sumcloud[2])
		max_sumcloud = max(sumcloud[0],sumcloud[1],sumcloud[2])

print("Matching: "    ,numpy.round(max_summatching,4))
print("One Edge: "    ,numpy.round(max_sumedge,4))
print("In Cloud: "   ,numpy.round(max_sumcloud,4))
print()
'''
