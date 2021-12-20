#!/bin/bash
# 
# arg1: mq_read

mq_read=${1}

# example output of kubetools when there is a message in the queue
#-----------------------------------------------------------------
# 2020/05/01 15:45:51 receiving queue message from hello-world-queue channel
# 2020/05/01 15:45:51 received 1 messages, 0 messages Expired 
# 2020/05/01 15:45:51 queue message received:
# 	this is a queue message 1
#-----------------------------------------------------------------

# to get the message, get a line next to a line having key words of 'queue message received' and trim the trailing indentation
message=$(kubetools queue receive "${mq_read}" 2>&1 | awk '/queue message received/{getline; print}' | awk '{$1=$1;print}')

echo ${message}


