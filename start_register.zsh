#!/bin/zsh
cd "$(dirname "$0")"
conda activate seiyufinder
python3 register.py "$@"
