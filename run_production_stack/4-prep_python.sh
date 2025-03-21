#!/bin/bash
sudo apt-get install -y python3.12-venv python3-pip
python3 -m venv /home/ubuntu/prodstack
. /home/ubuntu/prodstack/bin/activate

# Get the script directory to reference local scripts reliably.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../benchmarks/multi-round-qa/"
python3 -m pip install -r requirements.txt
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
mv ShareGPT_V3_unfiltered_cleaned_split.json ShareGPT.json
python3 cleanup_sharegpt.py $1 ShareGPT.json