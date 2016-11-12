#!/bin/sh

# Run the training using the pretrained model
../../caffe-sl/build/tools/caffe train -solver ../../models/baseline_activity/solver.prototxt -weights ../../models/baseline_activity/pretrained_model/bvlc_alexnet.caffemodel -gpu 0 2>&1 | tee runlogs/log.txt &
