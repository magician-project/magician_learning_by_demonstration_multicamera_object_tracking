#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Exit on error, unset vars, or pipe failure
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIR="$1"
EXT="csv"

PYTHON_SCRIPT="plot_csv.py"  # change to your python script name

if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist."
    exit 1
fi

shopt -s nullglob

source venv/bin/activate

for file in "$DIR"/*"$EXT"; do
    echo "Processing: $file"
    python3 "$PYTHON_SCRIPT" "$file" --headless --save_mp4 $file-plot.mp4

done

