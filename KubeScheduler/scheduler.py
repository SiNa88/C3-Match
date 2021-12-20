#!/usr/bin/env python

import time
import random
import json
import re
import subprocess
import sys
import os
import numpy
from kubernetes import client, config, watch
from prometheus_api_client import PrometheusConnect

start_time = time.monotonic()

prom = PrometheusConnect()

## grab latency between two nodes
#print(prom.custom_query(query="sum(abs(delta(download_durations_s_sum{}[1m]))< 50)by(source_node_name,dest_node_name)"))

config.load_kube_config()
v1=client.CoreV1Api()

scheduler_name = "discc"

def nodes_available():
    ready_nodes = []
    for n in v1.list_node().items:
            for status in n.status.conditions:
                if status.status == "True" and status.type == "Ready":
                    ready_nodes.append(n.metadata.name)
    #print("available nodes: ",ready_nodes)
    return ready_nodes

def choice(nodes):
    resources = nodes
    #print(resources)
    resources_str = "["+(" ".join(str(x) for x in resources)).replace(" ", ",") + "]"
    #print ((resources_str))
    tasks = ["encoding","framing","training"]
    #print((tasks))
    #"inference","lowAccuracy","highAccuracy","transcoding"]
    tasks_str = "["+(" ".join(str(x) for x in tasks)).replace(" ", ",") + "]"
    #for task in ((tasks)):
    command = str.encode(os.popen("python3.9 ranking_levelized.py "+tasks_str+" "+ resources_str).read())
    output = command.decode()
    candidates_nodes = ((("["+(output).replace("\n",",")+"]")).strip("][").split(","))[0:len(tasks)]
    #print (candidates_nodes)
    candidates_nodes_indexes = list(map(int, candidates_nodes))
    #print (len(candidates_nodes_indexes))
    candidates_nodes_names =  [0 for i in range(len(tasks))]
    for i in range(len(candidates_nodes)):
    	candidates_nodes_names[i] = nodes[candidates_nodes_indexes[i]]
    #print (candidates_nodes_names)
    #############command_dep_encod = str.encode(os.popen("kubectl apply -f ~/Documents/NaMe/project/0encoding/00encoding-deployment.yaml").read())
    #############command_dep_fram  = str.encode(os.popen("kubectl apply -f ~/Documents/NaMe/project/1framing/11framing-deployment.yaml").read())
    #############command_dep_train = str.encode(os.popen("kubectl apply -f ~/Documents/NaMe/project/3training/33training-deployment.yaml").read())
    ########query= prom.custom_query(query='sort_desc((container_cpu_user_seconds_total))')
    ###print("prometheus returned: ", query)
    ########node = query[0]['metric']['node']
    ###node = random.choice(nodes)
    #print("selected :", node)
    return candidates_nodes_names

def scheduler(pod, node, namespace="default"):
    # print("start binding")
    try:
        target = client.V1ObjectReference()
        target.kind = "Node"
        target.apiVersion = "v1"
        target.api_version = 'v1'
        target.name = node
        #print("Target object: ", target)
        if target.name != '':
            meta = client.V1ObjectMeta()
            meta.name = pod.metadata.name
            body = client.V1Binding(target=target, metadata=meta)
            v1.create_namespaced_binding(namespace, body, _preload_content=False)
            print(meta.name, " scheduled on ", node)
        #else:
        #   print(pod.metadata.name, " not scheduled")
    except client.rest.ApiException as e:
        print(json.loads(e.body)['message'])
        print("------------------------------------------")
    return


class Test(object):
    def __init__(self, data):
	    self.__dict__ = json.loads(data)

def main():
    w = watch.Watch()
    command_dep_encod = str.encode(os.popen("kubectl apply -f ~/Documents/NaMe/project/0encoding/00encoding-deployment.yaml").read())
    command_dep_fram  = str.encode(os.popen("kubectl apply -f ~/Documents/NaMe/project/1framing/11framing-deployment.yaml").read())
    command_dep_train = str.encode(os.popen("kubectl apply -f ~/Documents/NaMe/project/3training/33training-deployment.yaml").read())
    for event in w.stream(v1.list_namespaced_pod, "default"):
        ###regex = "\s*'status':\s*'(True)',\s*'type': '(PodScheduled)'"
        ###check = re.search(regex,str(event['object'].status.conditions))
        #print ("Hey there!")
        i = 0
        #print(event['object'])
        #print(v1.list_namespaced_pod(namespace='default', label_selector='.status.conditions'.format(name))
        if event['object'].status.phase == "Pending" and event['object'].spec.scheduler_name == scheduler_name:
            try:
                print("------------------------------------------")
                print("scheduling pod ", event['object'].metadata.name)
                res = scheduler(event['object'], choice(nodes_available())[i])
                #if i == 2:
                #     break
                #i = i + 1
                #print (i)
            except client.rest.ApiException as e:
                print (json.loads(e.body)['message'])

if __name__ == '__main__':
    main()
elapsed_time = numpy.round(time.monotonic() - start_time , 5)
print ("=====================================================================")
print ("Algorithm execution time: {} second(s)".format(elapsed_time))
print ("=====================================================================")
