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
	try:
		img=skimage.io.imread(img_dir+fl)
	except:
		print fl, 'does not exist?'
		return None

	try:
		img=np.rollaxis(img,2) #(3,256,256)
	except:
		#shutil.copyfile(img_dir+fl, 'failed/'+fl)
		return None

	return img 

def label_actions(fl):
	return class_dict[fl.split('_')[0]]


obj_labels=pickle.load(open('../imsitu/metadata/obj_labels.pkl','rb'))
def label_objects(fl): #max 30 labels? :\
	label_vec=np.zeros(30)-1
	label_set=obj_labels[fl]
	if len(label_set)>0:
		label_vec[0:len(label_set)]=np.array(list(label_set))
	return label_vec
#----------------------------------------------------------------#

def make_hdf5(filenames, out_file, train=False, shuffle=True):

	imgs=parallelize(read_img, filenames, 10)
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

	print img_data.shape

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
img_dir='../imsitu/of500_images_resized/'
subset_files=read_lines('../imsitu_subset/imsitu_subset.txt')
print len(subset_files)

with open('../imsitu/metadata/class_dict.txt','r') as f:
	class_dict=f.read().strip().split('\n')
	class_dict=[d.split('\t') for d in class_dict]
	class_dict=dict([[d[0], int(d[1])] for d in class_dict])

img_mean=pickle.load(open('imagenet_mean.pkl','rb'))
make_hdf5(subset_files, 'imsitu_subset/subset_data.h5', True)






