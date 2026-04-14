#!/bin/bash
# Activate playpen pytorch_env from the RL directory.
# Usage:  source activate_playpen_env.sh   (or  . activate_playpen_env.sh)
# Then run:  PYTHONPATH=$PWD python optical_rl/main_stage1_angled.py
#
# Note: pytorch_env currently has PyTorch CPU-only. For GPU training, in a shell
# where this is sourced run:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121

PLAYPEN_ENV="/playpen/hisham26/conda/envs/pytorch_env"
export PATH="$PLAYPEN_ENV/bin:$PATH"
export CONDA_PREFIX="$PLAYPEN_ENV"

# Optional: Node/nvm for ray-optics runner
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

echo "Playpen env active: $PLAYPEN_ENV (python: $(which python 2>/dev/null || which python3))"
