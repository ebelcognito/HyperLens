#!/bin/bash
#SBATCH -J oct_traces
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/nas/longleaf/home/hisham26/Desktop/hyperlens/logs/oct_traces_%j.log

set -e
cd /nas/longleaf/home/hisham26/Desktop/hyperlens

PY=/nas/longleaf/home/hisham26/miniforge3/envs/hyperlens/bin/python

echo "=== STORYBOARD PICK (70 designs, 3 wvln x 7 fields) ==="
STORYBOARD_WORKERS=24 $PY scripts_throwaway/pick_chromatic_storyboard.py

echo ""
echo "=== PRECOMPUTE (3837 designs, 3 wvln x 7 fields) ==="
PRECOMPUTE_WORKERS=30 $PY scripts_throwaway/precompute_octuplet_chromatic_traces.py

echo ""
echo "=== ALL DONE ==="
