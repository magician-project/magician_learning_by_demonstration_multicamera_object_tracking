#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

python3 -m venv venv
source venv/bin/activate

python3 -m pip install opencv-contrib-python reportlab



exit 0
