#!/bin/bash

set -euo pipefail
echo "MY_VAR1: $MY_VAR1"
echo "MY_VAR2: $MY_VAR2"
echo "PUBLIC_KEY: $PUBLIC_KEY"
nvidia-smi
. /start.sh
