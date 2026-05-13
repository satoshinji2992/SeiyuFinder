#!/bin/zsh
cd "$(dirname "$0")"
conda activate seiyufinder
nohup python3 server.py "$@" > output.log 2>&1 &
echo "Server started on port 3724, log: output.log"
