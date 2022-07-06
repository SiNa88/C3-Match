import numpy
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter
from scipy.stats.mstats import rankdata    
import networkx as nx
import yaml
import subprocess
import json
import sys
import os
sys.path.append('./diff_times')
from diff_times import comp_times


#tasks0 = ["encode_20000","frame_20000","hightrain","inference","encode_20000"]#sys.argv[1].strip("][").split(",")
#print(type(tasks0))

# Converting string to list
num_of_apps = len(sys.argv) - 2
lenn_tasks = 5 #max(len(tasks0) , len(tasks1))
#print(lenn_tasks)
#print(len(sys.argv))
levels = [[0  for i in range(lenn_tasks)] for j in range(num_of_apps)]
for i in range(1,len(sys.argv)-1):
	levels[i-1] = sys.argv[i].strip("][").split(",")
	#print (i)

#              0	       1	      2      		3      4        5        6       7		8
#resources = ["vm-bg","vm-exo-med","vm-exo-lg","egs","lenovo","jetson","rpi4","rpi3"]
resources = sys.argv[len(sys.argv)-1].strip("][").split(",")


lat =	[[ 0.5e-3, 26e-3, 15e-3 , 22.4e-3 , 22.8e-3 , 22.8e-3 , 22.8e-3 , 22.8e-3],
		[ 26e-3,  0.5e-3 , 12.5e-3 , 18.4e-3 , 18.4e-3 , 18.4e-3 , 18.4e-3 , 18.4e-3],
		[ 15e-3 , 12.5e-3 , 0.5e-3  , 7.2e-3 , 7.2e-3 , 7.5e-3 , 7.5e-3 , 7.5e-3], 
		[ 22.4e-3, 18e-3 , 7.2e-3,  0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3], 
		[ 22.8e-3, 18.4e-3, 7.2e-3, 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3], 
		[ 22.8e-3, 18.4e-3, 7.5e-3, 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3],  
		[ 22.8e-3, 18.4e-3 , 7.5e-3, 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3], 
		[ 22.8e-3, 18.4e-3 , 7.5e-3, 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3 , 0.5e-3]]

T =  [[[[0]  for k in range(len(resources))] for i in range(lenn_tasks)] for j in range(num_of_apps)] 
Tm = [[[[0]  for k in range(len(resources))] for i in range(lenn_tasks)] for j in range(num_of_apps)] 
Tr = [[[[0]  for k in range(len(resources))] for i in range(lenn_tasks)] for j in range(num_of_apps)] 
Tq = [[[[0]  for k in range(len(resources))] for i in range(lenn_tasks)] for j in range(num_of_apps)] # Queuing of Data cells.

dictlistResources = list( {} for i in range(len(resources)) )
sorted_dictlistResources = list( {} for i in range(len(resources)) )
dictlisttasks = list( {} for i in range(lenn_tasks) )
sorted_dictlisttasks = list( {} for i in range(lenn_tasks) )


alloc = [[0  for j in range(lenn_tasks)] for i in range(num_of_apps)] 
for i in range(lenn_tasks):
	fileObject = open("D:\\00Research\\matching\\scheduler\\paper\\MoreApps\\MPL.yaml", 'w').close()
	capfile = open('D:\\00Research\\matching\\scheduler\\paper\\MoreApps\\capacities-testbed.yml', 'w').close()
	fileObject = open("D:\\00Research\\matching\\scheduler\\paper\\MoreApps\\DPL.yaml",'w').close()
	for k in range(num_of_apps):
		#print(levels[k][i])
		dictlisttasks = list( {} for i in range(lenn_tasks) )
		sorted_dictlisttasks = list( {} for i in range(lenn_tasks) )
		if (i != 0):
			Tm[k][i],Tr[k][i],Tq[k][i],T[k][i] = comp_times(resources, levels[k][i],alloc[k][i-1])
		else:
			Tm[k][i],Tr[k][i],Tq[k][i],T[k][i] = comp_times(resources, levels[k][i],4) 
		for j in range(len(resources)):
			#print(resources[j])
			dictlisttasks[i][(resources[j],levels[k][i])] = Tm[k][i][j]
			#print(levels[k][i],"  ",resources[j],"  ",Tm[k][i][j])
		sorted_dictlisttasks[i]=sorted(dictlisttasks[i].items(),key = itemgetter(1), reverse=False)
		#print((sorted_dictlisttasks[i]))
		#print(dict(sorted_dictlisttasks[i]).keys())
		mpllll=dict(dict(sorted_dictlisttasks[i]).keys())
		######################print(mpllll)
		listofvalues = list(mpllll.keys())
		listofkeys=list(mpllll.values())
		#print(listofvalues)
		dicttttt= {listofkeys[0]:listofvalues}
		#print((listofkeys[0]))
		with open(r'D:\\00Research\\matching\\scheduler\\paper\\MoreApps\\MPL.yaml', 'a') as file:
			documents = yaml.dump(dicttttt, file)
			file.close()
		#print(dict(sorted_dictlistResources[j]))
		#print()
		####print (sorted_dictlisttasks[i])
		####print()
		#for k in range(num_of_apps):
	#print()
	#print()
	dictlistResources = list( {} for i in range(len(resources)) )
	sorted_dictlistResources = list( {} for i in range(len(resources)) )
	for j in range(len(resources)):
				for k in range(num_of_apps):
					#print(levels[k][i],"  ",resources[j],"  ",Tr[k][i][j])
					dictlistResources[j][(levels[k][i],resources[j])] = numpy.round(numpy.round(Tr[k][i][j],4),4)
					sorted_dictlistResources[j] = sorted(dictlistResources[j].items(),key = itemgetter(1), reverse=True)
					dpllll = dict(dict(sorted_dictlistResources[j]).keys())
					#print(k," ", dpllll)
				######################print (dpllll)
				listofvalues = list(dpllll.keys())
				listofkeys = list(dpllll.values())
				#print(listofvalues)
				dicttttt = {listofkeys[0]:listofvalues}
				#print((listofkeys[0]))
				with open(r'D:\\00Research\\matching\\scheduler\\paper\\MoreApps\\DPL.yaml', 'a') as file:
					documents = yaml.dump(dicttttt, file)
					file.close()
				####print(dict(sorted_dictlistResources[j]))
				####print()
				capacity_dict = {listofkeys[0]:2}
				with open(r'D:\\00Research\\matching\\scheduler\\paper\\MoreApps\\capacities-testbed.yml', 'a') as capfile:
					yaml.dump(capacity_dict, capfile)
					file.close()
	#print()
	########################-----------------------------------------------------------------------------------------------------------------------------################
	command = str.encode(os.popen("C:\\Users\\narmehran\\AppData\\Local\\Programs\\Python\\Python39\\python.exe "+"D:\\00Research\\matching\\scheduler\\paper\\MoreApps\\matchingalgo-testbed.py").read())
	output = command.decode()
	with open('D:\\00Research\\matching\\scheduler\\paper\\MoreApps\\matching-testbed.yaml', 'r') as file_read:
		# The FullLoader parameter handles the conversion from YAML
		# scalar values to the dictionary format
		matching_list = yaml.load(file_read, Loader=yaml.FullLoader)
		#print()
		#print((matching_list))
		file_read.close()
		dataa = json.loads(matching_list)
		for ind in range(len(dataa)):
			#print((dataa[ind]))
			#print()
			for key, values in dataa[ind].items():
					#print(key, "    ", (values))
					#print(resources.index(key))
					for value in ((values)):
						for appp in range(num_of_apps):
							for task in levels[appp]:
								#print(task, " ", value)
								if (value == (task)):
									#print(levels[appp].index(task))
									#print(task)
									alloc[appp][levels[appp].index(task)] = resources.index(key)
									#print(resources.index(key))
	#print(":D    :D    :D")
print (alloc)