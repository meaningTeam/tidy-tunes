#!/bin/bash

INPUT_DIR=${1}
OUTPUT_DIR=${2}
PARTITIONS=${3:-1}
CONFIG_PATH=${4:-configs/full_en.yaml}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "partition_00" ]]; then
    echo "Generating partitions..."
    
    # Find, shuffle, and split the files into partitions
    find "$INPUT_DIR" -type f -name "*.flac" | shuf > all_files.txt
    split -d -n l/"$PARTITIONS" all_files.txt partition_
    rm all_files.txt
fi

for idx in $(seq -f "%02g" 0 $(($PARTITIONS - 1))); do
    part_file="partition_$idx"
    sbatch \
        -p a100 \
        --gpus=1 \
        --cpus-per-gpu=8 \
        --job-name=audio_proc_$idx \
        --output=slurm.audio_proc_$idx.out \
        --export=ALL,PART_IDX=$idx,PART_FILE=$part_file,OUTPUT_DIR=$OUTPUT_DIR,CONFIG_PATH=$CONFIG_PATH \
	"$SCRIPT_DIR/run_job.sh"
done