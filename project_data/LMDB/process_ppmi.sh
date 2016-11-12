#!/usr/bin/env sh
# Obtain the lmdb label text files
echo 'Deleting previous generations'
rm -rf ppmi/*lmdb

# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

DATA=utils/lmdb_text
EXAMPLE=ppmi
TOOLS=../../caffe-sl/build/tools

PPMI_TRAIN_DATA_ROOT=../images/ppmi/
PPMI_TEST_DATA_ROOT=../images/ppmi/
# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi


echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $PPMI_TRAIN_DATA_ROOT \
    $DATA/ppmi_train.txt \
    $EXAMPLE/train_lmdb

echo "Creating test lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $PPMI_TEST_DATA_ROOT \
    $DATA/ppmi_test.txt \
    $EXAMPLE/test_lmdb

echo "Done."
