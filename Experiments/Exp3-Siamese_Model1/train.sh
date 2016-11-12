#!/bin/sh

# Run the training using the pretrained model
../../caffe-sl/build/tools/caffe train -solver ../../models/object-activity/solver.prototxt -weights ../../models/object-activity/pretrained_model/bvlc_dissimilarity.caffemodel -gpu 0 2>&1 | tee runlogs/log.txt &
