#!/bin/bash
mkdir third_party/KP2D/data
mkdir third_party/KP2D/data/models
mkdir third_party/KP2D/data/models/kp2d/ 
wget https://tri-ml-public.s3.amazonaws.com/github/kp2d/models/pretrained_models.tar.gz
mv pretrained_models.tar.gz third_party/KP2D/data/models/kp2d/
cd third_party/KP2D/data/models/kp2d/ && tar -xzf pretrained_models.tar.gz 
