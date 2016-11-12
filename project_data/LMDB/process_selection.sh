#!/usr/bin/env sh
# Obtain the lmdb label text files
echo 'Deleting previous generations'
rm -rf utils/lmdb_text/imsitu_selection.txt
rm -rf imsitu_selection/*lmdb

echo 'Generating label files'
python imsitu_selection/labeller.py utils/images_info/imsitu_selection.txt utils/lmdb_text/imsitu_selection.txt utils/label_map/activity_label.p
echo 'File generation done.. moving forward...'
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

DATA=utils/lmdb_text
EXAMPLE=imsitu_selection
TOOLS=../../caffe-sl/build/tools

SELECTION_DATA_ROOT=../images/imsitu/

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
    $SELECTION_DATA_ROOT \
    $DATA/imsitu_selection.txt \
    $EXAMPLE/selection_lmdb

echo "Done."
