#Instructions to run it

python test_extractor.py <grounded/ungrounded pickle file> <The multiplier above mean for threshold> <The number of activities in which the object is least present>

Eg - python test_extractor.py obj_labels_imagenet_grounded.pkl 3 3

# The result images are the images which are corresponding to activities in which the object is least(k-least) present, for all objects that are heavily biased towards at least one activity
test_images.txt
