#!/bin/bash

# Output some informative diagnostics
echo '#' Running on $(hostname)
echo '#' Started at $(date)
set | grep SLURM | while read line; do echo "# $line"; done
echo -n '# '; cat <<EOF
$@
EOF

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR/$PART_IDX"

# Process the files using Python script
tidytunes process-audios -c "$CONFIG_PATH" -o "$OUTPUT_DIR/$PART_IDX" -d cuda `cat "$PART_FILE"`