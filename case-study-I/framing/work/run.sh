#!/bin/bash 
# arg1: input file
# arg2: output directory

input_file=${1}
input_directory=${2}
#output_directory=${3}
#mq_write=${4}

echo '#'
echo '#  Starting Process: '
echo '#'

START=$(date +%s)
python split.py "${input_file}" "${input_directory}"

ls ${input_directory}
END=$(date +%s)
DIFF=$(( $END - $START )) 
echo "It took $DIFF seconds" && exit
echo '	Done'

