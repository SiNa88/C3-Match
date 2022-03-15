#!/bin/bash
# arg1: work_path
# arg2: input directory
# arg3: work directory
# arg4: output directory
# arg5: mq_read
# arg6: mq_write

#work_path=${1}
#input_directory=${2}
#work_directory=${3}
#output_directory=${4}
#mq_write=${6}


echo '#'
echo '#  Starting Process: Unpacking '
echo '#'

# Extract tar.gz file
echo "   Extracting ..."
START=$(date +%s%3N)

tar -xvzf amazon_review_polarity_csv.tar.gz
END=$(date +%s%3N)
DIFF=$(( $END - $START ))
echo "It took $DIFF milliseconds"

echo '   Done'