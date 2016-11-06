# Folder Descriptions

* imsitu --> raw imsitu images
	* images_info: .txt files of filenames for train/dev/test split
	* label_map: contains the dictionary pickle that maps all the activities to actions #No need for this folder. We're both using alphabetical ordering	
	* object_labels: object labels for imsitu
	* lmdb_text --> Contains the image label pair .txt files that will be used by caffe to create the lmdb files

* ppmi --> raw ppmi images

* imsitu_selections --> The directory in which we will store all the codes of the curated imsitu test set (the one which had originally 1034 images)

* hdf5 --> directory with hdf5 files of all datasets (imsitu/ppmi/imsitu_selections) and scripts to generate them

