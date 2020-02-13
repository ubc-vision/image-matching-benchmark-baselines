#!/bin/bash

# D2-net
echo "Extracting D2-net"
python extract_d2net.py --method_name=d2net-singlescale_8000 --num_kp=8000
python extract_d2net.py --multiscale --method_name=d2net-multiscale_8000 --num_kp=8000
