import glob
import h5py
import numpy as np
import skimage.io
from sklearn.utils import shuffle as sk_shuffle
from joblib import Parallel, delayed
import cPickle as pickle

# treat PPMI as a two class dataset (plays/does not) across all instruments. Save data to hdf5

#------------------------------util------------------------------#

def parallelize(func, itemlist, verbose=1):
	return Parallel(n_jobs=20, verbose=verbose)(delayed(func)(item) for item in itemlist)

def read_img(fl): #skimage gives you an RGB, cv2 gives BGR
	img=skimage.io.imread(fl)
	img=np.rollaxis(img,2) 
	
	#crop the one extra pixel off
	img=img[:,1:-1,1:-1] #(3,258,258) -> (3,256,256) 

	return img 

def label_objects(fl):
	if 'with_instrument' in fl:
		return 0
	elif 'play_instrument' in fl:
		return 1
	return -1
	
def write_hdf5(data, out_file):
	with h5py.File(out_file, 'w') as hf:
		_, img_data, labels=zip(*data)
		img_data, labels=np.asanyarray(img_data), np.array(labels)
		hf.create_dataset('images', data=img_data)
		hf.create_dataset('labels', data=labels)
		hf.create_dataset('img_mean', data=img_mean)

#----------------------------------------------------------------#

filenames=glob.glob('ppmi/*_instrument/*/*/*.jpg')
imgs=parallelize(read_img, filenames, 10)
labels=parallelize(label_objects, filenames)
data=zip(filenames, imgs, labels)

train_data=[d for d in data if '/train/' in d[0]]
test_data=[d for d in data if '/test/' in d[0]]
train_data=sk_shuffle(train_data) #shuffle only train
img_mean=pickle.load(open('imagenet_mean.pkl','rb'))

write_hdf5(train_data, 'hdf5/ppmi/train_data.h5')
write_hdf5(test_data, 'hdf5/ppmi/test_data.h5')
