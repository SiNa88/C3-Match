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


#https://aws.amazon.com/kinesis/data-firehose/pricing/?nc=sn&loc=3

#seg_size = 80000 #(10KB)
#video_size = 8000000000 #(1GB)
index_of_segment = 4
seg_size = [286720, 2457600, 3440640, 14400000, 20971520 ] #bits
video_size = [2000000, 14000000, 28000000, 60000000, 204800000] #bits

#tasks0 = sys.argv[1].strip("][").split(",")
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



T =  [[[[0]  for k in range(len(resources))] for i in range(len(tasks0))] for j in range(num_of_apps)] 
Tm = [[[[0]  for k in range(len(resources))] for i in range(len(tasks0))] for j in range(num_of_apps)] 
Tr = [[[[0]  for k in range(len(resources))] for i in range(len(tasks0))] for j in range(num_of_apps)] 
Tq = [[[[[0] for j in range(len(resources))] for k in range(10)] for i in range(len(tasks0))] for l in range(num_of_apps)] # Queuing of Data cells.

#T = [[0] * len(resources) for i in range(len(tasks0))]
#Tm = [[0] * len(resources) for i in range(len(tasks0))]
#Tr = [[0] * len(resources) for i in range(len(tasks0))]
#Tq = [[[[0] for j in range(len(resources))] for k in range(10)] for i in range(len(tasks0))] # Queuing of Data cells.

for i in range(num_of_apps):
	Tm[i],Tr[i],Tq[i],T[i] = comp_times()
# Tm[i][:][:],Tr[i][:][:],Tq[i][:][:][:],T[i][:][:]
# One paper had considered #VMs : 60 Cloud - 20 Cloudlet - 10 Edge


#0.015 - 0.8 ms
#65 - 85 ms
#Cloud, Tier2, Tier1(Vienna), Barcelona, Amsterdam, Paris, Brussels, Frankfurt, Graz, Ljubljana, London, Stockholm, Vienna 
#		0  1  2  3  4  5  6 7  8  9 10 11 12


# To be comparable with bw and cpu ratios.
#lambda_in = [40,40,40,40,40]
#lambda_out = [40,40,40,40,40]


dictlistResources = list( {} for i in range(len(resources)) )
sorted_dictlistResources = list( {} for i in range(len(resources)) )
dictlisttasks = list( {} for i in range(len(tasks0)*num_of_apps) )
sorted_dictlisttasks = list( {} for i in range(len(tasks0)*num_of_apps) )

#Earliest start time
#EST = [[[[0] for j in range(len(resources))] for k in range(10)] for i in range(len(tasks0))]


mytpl = dict()
#print()
#print()
fileObject = open("D:\\00Research\\matching\\scheduler\\paper\\TPL.yaml", 'w').close()
for k in range(num_of_apps):
	for i in range(len(tasks0)):
		for j in range(len(resources)):		
			dictlisttasks[i+(k*len(tasks0))][(resources[j],tasks0[i]+str(k))] = numpy.round(numpy.round(Tm[k][i][j],4) + numpy.round(Tq[k][i][1][j],4), 4)
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
myrpl = dict()
capfile = open('D:\\00Research\\matching\\scheduler\\paper\\capacities-testbed.yml', 'w').close()
fileObject = open("D:\\00Research\\matching\\scheduler\\paper\\RPL.yaml",'w').close()
#print (sorted(dictlisttasks.items(),key = itemgetter(1)))
for j in range(len(resources)):
	for k in range(num_of_apps):
		for i in range(len(tasks0)):
			dictlistResources[j][(tasks0[i]+str(k),resources[j])] = numpy.round(numpy.round(Tr[k][i][j],4),4) #numpy.round(Tm[k][i][j],4)+numpy.round(Tq[k][i][1][j],4)+
	sorted_dictlistResources[j]=sorted(dictlistResources[j].items(),key = itemgetter(1), reverse=True)
	rpllll=dict(dict(sorted_dictlistResources[j]).keys())
	#myrpl = rpllll
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



#[(('highAccuracy', 't-1'), 0), (('analysis', 't-1'), 0), (('transcoding', 't-1'), 0), (('packaging', 't-1'), 0), (('snk', 't-1'), 0), (('src', 't-1'), 0.2236), (('framing', 't-1'), 1.3697), (('lowAccuracy', 't-1'), 1.3697)]

alloc = [0 for i in range(len(newtasks))]

popen = subprocess.Popen(["C:\\Users\\narmehran\\AppData\\Local\\Programs\\Python\\Python39\\python.exe" ,"D:\\00Research\\matching\\scheduler\\paper\\matchingalgo-testbed.py"], stdout=subprocess.PIPE)
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

'''with open(r'D:\\00Research\\matching\\scheduler\\paper\\TPL.yaml') as tplllll:
	tpl_list = yaml.load(tplllll, Loader=yaml.FullLoader)
	#print((matching_list))
	dataa = json.loads(tpl_list)
'''


'''
with open(r'E:\\data\\fruits.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    fruits_list = yaml.load(file, Loader=yaml.FullLoader)

    print(fruits_list)

dict_file = [
	{'sports' : ['soccer', 'football', 'basketball', 'cricket', 'hockey', 'table tennis']},
	{'countries' : ['Pakistan', 'USA', 'India', 'China', 'Germany', 'France', 'Spain']}
	]
with open(r'store_file.yaml', 'w') as file:
    documents = yaml.dump(dict_file, file)'''


'''with open(r'matching-testbed.yaml') as file_read:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    matching_list = yaml.load(file_read, Loader=yaml.FullLoader)
    #print((matching_list))
    daddee = json.loads(matching_list)
    for i in range (len(daddee)):
    	print ((daddee[i]))'''
#print(())
#print()
#dictttt = dict(matching_list)