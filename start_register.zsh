#!/bin/zsh
cd "$(dirname "$0")"
source venv/bin/activate
python3 register.py "$@"
