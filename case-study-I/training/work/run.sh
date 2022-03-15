#!/bin/bash 
# arg1: input file
# arg2: output directory

#input_file=${1}
input_directory=${1}
#output_directory=${3}
#mq_read=${2}
#mq_write=${3}

echo '#'
echo '#  Starting Process: '
echo '#'

python3 Traffic_sign_classification-checkpoint-highaccuracy.py
