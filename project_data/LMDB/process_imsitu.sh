#!/usr/bin/env sh
# Obtain the lmdb label text files
echo 'Deleting previous generations'
rm -rf utils/lmdb_text/imsitu_train.txt utils/lmdb_text/imsitu_val.txt utils/lmdb_text/imsitu_test.txt
rm -rf imsitu/*lmdb

echo 'Generating label files'
python imsitu/labeller.py utils/images_info/imsitu_train.txt utils/images_info/imsitu_val.txt utils/images_info/imsitu_test.txt utils/lmdb_text/imsitu_train.txt utils/lmdb_text/imsitu_val.txt utils/lmdb_text/imsitu_test.txt utils/label_map/activity_label.p
echo 'File generation done.. moving forward...'
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

DATA=utils/lmdb_text
EXAMPLE=imsitu
TOOLS=../../caffe-sl/build/tools

TRAIN_DATA_ROOT=../images/imsitu/
VAL_DATA_ROOT=../images/imsitu/
TEST_DATA_ROOT=../images/imsitu/

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

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

if [ ! -d "$TEST_DATA_ROOT" ]; then
  echo "Error: TEST_DATA_ROOT is not a path to a directory: $TEST_DATA_ROOT"
  echo "Set the TEST_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/imsitu_train.txt \
    $EXAMPLE/train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/imsitu_val.txt \
    $EXAMPLE/val_lmdb

echo "Creating test lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TEST_DATA_ROOT \
    $DATA/imsitu_test.txt \
    $EXAMPLE/test_lmdb

echo "Done."
