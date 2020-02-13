#!/bin/bash

# superpoint
echo "Extracting Superpoint"
python third_party/superpoint_forked/superpoint.py --cuda --num_kp=2048 --nms_dist=4 --resize_image_to=1200 --method_name=superpoint-nms4-r1200-lowerdet --conf_thresh=0.0001
python third_party/superpoint_forked/superpoint.py --cuda --num_kp=2048 --nms_dist=3 --resize_image_to=1200 --method_name=superpoint-nms3-r1200-lowerdet --conf_thresh=0.0001
python third_party/superpoint_forked/superpoint.py --cuda --num_kp=2048 --nms_dist=2 --resize_image_to=1200 --method_name=superpoint-nms2-r1200-lowerdet --conf_thresh=0.0001
python third_party/superpoint_forked/superpoint.py --cuda --num_kp=2048 --nms_dist=1 --resize_image_to=1200 --method_name=superpoint-nms1-r1200-lowerdet --conf_thresh=0.0001
