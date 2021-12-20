#!/bin/bash 
# arg1: input file
# arg2: output directory

#input_file=${1}
input_directory=${1}
#output_directory=${3}
mq_read=${2}
mq_write=${3}

echo '#'
echo '#  Starting Process: '
echo '#'

python3 Traffic_sign_classification-checkpoint-highinference.py


# Loop through all files in argument list
for from_file_path in $(ls ${input_directory}/*.jpg)
do
    curr_file_name=${from_file_path##*/} 
    echo "	${curr_file_name}"
    #to_file_path=${output_directory}

    #cp ${from_file_path} ${to_file_path}

    # Write to message queue
    if [  "${mq_read}" != "-" ]; then
        echo "	Reading file name ${curr_file_name} from a message queue ${mq_read}"
        result="$(./read_from_mq.sh ${mq_read} ${curr_file_name})"
	  echo ${result} 
    fi 
done
