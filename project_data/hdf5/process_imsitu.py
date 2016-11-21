# script to preprocess the image and save in hdf5.
import cPickle as pickle
import h5py
import numpy as np
import skimage.io
from sklearn.utils import shuffle as sk_shuffle
from joblib import Parallel, delayed
import os.path

#------------------------------util------------------------------#
def remove_idx(L, null_idx):
	return [v for i, v in enumerate(L) if i not in null_idx]

def read_lines(flname):
	with open(flname,'r') as f:
		return f.read().strip().split('\n')

def parallelize(func, itemlist, verbose=1):
	return Parallel(n_jobs=20, verbose=verbose)(delayed(func)(item) for item in itemlist)

def read_img(fl): #skimage gives you an RGB, cv2 gives BGR
	img=skimage.io.imread(img_dir+fl)
	try:
		img=np.rollaxis(img,2) #(3,256,256)
	except:
		#shutil.copyfile(img_dir+fl, 'failed/'+fl)
		return None

	return img 

def label_actions(fl):
	return class_dict[fl.split('_')[0]]


obj_labels=pickle.load(open('imsitu/obj_labels.pkl','rb'))
def label_objects(fl): #max 30 labels? :\
	label_vec=np.zeros(30)-1
	label_set=obj_labels[fl]
	if len(label_set)>0:
		label_vec[0:len(label_set)]=np.array(list(label_set))
	return label_vec
#----------------------------------------------------------------#

def make_hdf5(filenames, out_file, train=False, shuffle=True):

	imgs=parallelize(read_img, filenames, 2)
	act_labels=parallelize(label_actions, filenames)
	obj_labels=parallelize(label_objects, filenames)

	# remove junk images (and corresponding labels)	
	junk_idx=set([i for i in range(len(imgs)) if imgs[i] is None])
	imgs = remove_idx(imgs, junk_idx)
	img_files = remove_idx(filenames, junk_idx)
	act_labels= remove_idx(act_labels, junk_idx)
	obj_labels= remove_idx(obj_labels, junk_idx)

	if shuffle:
		imgs, img_files, act_labels, obj_labels=sk_shuffle(imgs, img_files, act_labels, obj_labels)

	img_data=np.asanyarray(imgs)
	act_labels=np.array(act_labels)
	obj_labels=np.array(obj_labels)
	img_files=np.array(img_files)

	with h5py.File(out_file, 'w') as hf:
		hf.create_dataset('image_files', data=img_files)
		hf.create_dataset('images', data=img_data)
		hf.create_dataset('obj_labels', data=obj_labels)
		hf.create_dataset('act_labels', data=act_labels)
		hf.attrs['num_acts']=len(class_dict) #imsitu
		hf.attrs['num_objs']=1000 #imagenet

		if train:		
			hf.create_dataset('img_mean', data=img_mean)



#--------------------------------------------------------------------#

#mean_file='../models/pretrained/imagenet_mean.binaryproto'
#if not os.path.isfile('imagenet_mean.pkl'):

#	import sys
#	sys.path.append('/work/04340/tushar_n/packages/caffe/python')
#	import caffe
#	caffe.set_device(0)
#	
#	blob = caffe.proto.caffe_pb2.BlobProto()
#	data = open(mean_file,'rb').read()
#	blob.ParseFromString(data)
#	mean_val = np.array(caffe.io.blobproto_to_array(blob))[0] #BGR
#	mean_val = mean_val[::-1,:,:] #RGB
#	pickle.dump(mean_val, open('imagenet_mean.pkl','wb')) #(3,256,256)
#	print 'imagenet mean np file generated'

img_dir='../images/imsitu/of500_images_resized/'
train_files=read_lines('imsitu/train.txt')
val_files=read_lines('imsitu/dev.txt')
test_files=read_lines('imsitu/test.txt')

class_dict=list(set([fl.split('_')[0] for fl in train_files+val_files+test_files]))
class_dict=sorted(class_dict)
class_dict={cl:idx for idx, cl in enumerate(class_dict)}
with open('imsitu/class_dict.txt','w') as f:
	for cl, idx in sorted(class_dict.items(), key=lambda x: x[1]):
		f.write('%s\t%d\n'%(cl, idx))
print len(class_dict)

img_mean=pickle.load(open('imagenet_mean.pkl','rb'))
make_hdf5(train_files, 'imsitu/train_data.h5', True)
make_hdf5(val_files, 'imsitu/val_data.h5', False)
make_hdf5(test_files, 'imsitu/test_data.h5', False)







