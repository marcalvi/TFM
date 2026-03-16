#!/bin/bash

set -euo pipefail

echo "MY_VAR1: $MY_VAR1"
echo "MY_VAR2: $MY_VAR2"
#nvidia-smi
. /home/osiris-user/geb-activate.sh
conda env list
