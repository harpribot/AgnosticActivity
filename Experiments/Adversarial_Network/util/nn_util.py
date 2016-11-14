import os
import h5py
import numpy as np

def delete_file(filename):
	#deletes a file if it exsits 
	try:
		os.remove(filename)
	except OSError:
		pass
#--------------------------Data Loader--------------------------------#

#imsitu data
def load_imsitu(data_dir):
	train_data=h5py.File('%s/train_data.h5'%data_dir)
	val_data=h5py.File('%s/val_data.h5'%data_dir)

	img_mean= np.array(train_data['img_mean'])

	train_imgs, val_imgs=train_data['images'], val_data['images']
	train_files, val_files=train_data['image_files'], val_data['image_files']
	train_labels=(train_data['act_labels'], train_data['obj_labels'])
	val_labels=(val_data['act_labels'], val_data['obj_labels'])

	return [train_files, train_imgs, train_labels],\
			[val_files, val_imgs, val_labels],\
			img_mean

#ppmi data
def load_ppmi(data_dir):
	train_data=h5py.File('%s/train_data.h5'%data_dir)
	val_data=h5py.File('%s/test_data.h5'%data_dir)
	img_mean= np.array(train_data['img_mean'])
	return train_data['images'], train_data['labels'],\
			val_data['images'], val_data['labels'], img_mean


#agnostic imsitu subset
def load_subset(data_dir):
	val_data=h5py.File('%s/subset_data.h5'%data_dir)

	val_imgs=val_data['images']
	val_files=val_data['image_files']
	val_labels=(val_data['act_labels'], val_data['obj_labels'])
	img_mean= np.array(val_data['img_mean'])

	return [val_files, val_imgs, val_labels], img_mean



#--------------------------Batch Loader---------------------------#

def center_crop(img, crop_dims=(227,227)):
	"""Center crop an image according to crop_dims"""
	#h, w= img.shape[2:] # img should be (b,c,h,w)
	#pad_u, pad_l=(h-crop_dims[0])/2, (w-crop_dims[1])/2
	#pad_d, pad_r=(h-crop_dims[0])-pad_u, (w-crop_dims[1])-pad_l
	return img[:,:,14:-15,14:-15]

def vectorize_labels(labels, num_classes=1000, crit='BCE'):

	if crit=='NLL':
		label_vec=[]
		for idx, labs in enumerate(labels):
			if len(labs[labs>=0])==0:
				label_vec.append(-1)
			else:
				label_vec.append(np.random.choice(labs[labs>=0]))
		label_vec=np.array(label_vec)+1.0 # lua <3
		return label_vec

	if crit=='BCE':
		label_vec=np.zeros((len(labels), num_classes))
		for idx, labs in enumerate(labels):
			label_vec[idx][labs[labs>=0].astype(int)]=1
		return label_vec		

	return None


def get_random_batch(data, opt):

	_, images, labels= data

	act_labels, obj_labels= labels
	rand_idx=np.random.choice(images.shape[0], opt.batch_size, replace=False)
	rand_idx=sorted(rand_idx) #h5py is weird
	batch_data= np.subtract(images[rand_idx], opt.img_mean)	
	batch_data= center_crop(batch_data)

	act_lab_vec=act_labels[rand_idx]+1
	obj_lab_vec=vectorize_labels(obj_labels[rand_idx])
	batch_labels=[1.0*act_lab_vec, 1.0*obj_lab_vec]

	return 1.0*batch_data, batch_labels

def get_batches_in_sequence(data, opt):
	
	files, images, labels= data
	act_labels, obj_labels= labels

	#if required, pad with zeros
	if opt.eval==1:
		n_pad=opt.batch_size-images.shape[0]%opt.batch_size
		if n_pad>0:
			pad=np.zeros((n_pad, images.shape[1], images.shape[2], images.shape[3]))
			images=np.vstack([images, pad])
			act_labels=np.hstack([act_labels, np.zeros(n_pad)])
			obj_labels=np.vstack([obj_labels, np.zeros((n_pad, obj_labels.shape[1]))])
			files=np.hstack([files, np.zeros(n_pad)])

	for i in xrange(images.shape[0]/opt.batch_size):
		st, end= opt.batch_size*i, opt.batch_size*i + opt.batch_size
		batch_data= np.subtract(images[st:end], opt.img_mean)
		batch_data= center_crop(batch_data)

		act_lab_vec= act_labels[st:end]+1
		obj_lab_vec= vectorize_labels(obj_labels[st:end])
		batch_labels=[1.0*act_lab_vec, 1.0*obj_lab_vec]
		
		if opt.eval!=1:
			yield 1.0*batch_data, batch_labels

		else:
			batch_files= files[st:end]
			yield batch_files, 1.0*batch_data, batch_labels



#-----------------------------------------------------------------#
#ppmi, only val
def get_batches_in_sequence_ppmi(data, labels, opt):

	for i in xrange(data.shape[0]/opt.batch_size):
		st, end= opt.batch_size*i, opt.batch_size*i + opt.batch_size
		batch_data= np.subtract(data[st:end], opt.img_mean)
		batch_data= center_crop(batch_data)
		batch_labels= labels[st:end]+1

		yield 1.0*batch_data, 1.0*batch_labels

#-----------------------------------------------------------------#







