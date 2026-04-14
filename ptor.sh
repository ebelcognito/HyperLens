#!/bin/bash
# Quick shortcut to activate pytorch_env from playpen + Node.js via nvm
export PATH="/playpen/hisham26/conda/envs/pytorch_env/bin:$PATH"
export CONDA_PREFIX="/playpen/hisham26/conda/envs/pytorch_env"
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
exec $SHELL -i
