#!/usr/bin/env bash
# simple launcher
set -e
CONFIG=${1:-configs/model/lgbm.yaml}
python main.py train --config ${CONFIG}
