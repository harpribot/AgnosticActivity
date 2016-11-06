import h5py
import numpy as np
from util.nn_util import *
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.svm import SVC

#don't worry, we'll put it back in there
#model_dir='../../models/adversarial_object_activity'
model_dir='models'
data_dir='../../project_data'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--data_dir', default='%s/hdf5/ppmi'%data_dir)
parser.add_argument('--t7', default='nil')
parser.add_argument('--prototxt', default='nil')
parser.add_argument('--caffemodel', default='nil')
parser.add_argument('--final_layer', default='fc8') #'fc8_p'/'fc8-new'
opt = parser.parse_args()

#----------------------------------------------------------------#
def get_reps(inputs, labels, net):

	output_reps, output_labels=[], []
	for batchInputs, batchLabels in get_batches_in_sequence_ppmi(inputs, labels, opt):

		if opt.backend=='torch':
			net.predict(batchInputs)
			batchPred= net.get_reps().asNumpyTensor()
		elif opt.backend=='caffe':
			net.blobs['data'].data[...] = batchInputs[:,::-1,:,:]
			net.forward()
			batchPred = net.blobs[opt.final_layer].data.copy()

		output_reps.append(batchPred)
		output_labels.append(batchLabels)

	output_reps=np.vstack(output_reps)
	output_labels=np.hstack(output_labels)	
	return output_reps, output_labels #keep classes as 1,2


def torch_net():

	import PyTorchHelpers
	opt.load=opt.t7 #so that TorchTrainer can load
	TorchTrainer = PyTorchHelpers.load_lua_class('%s/AdversaryTrainer.lua'%model_dir, 'AdversaryTrainer')
	net = TorchTrainer(vars(opt))
	return net

def caffe_net():
	
	import sys
	sys.path.append('/work/04340/tushar_n/packages/caffe-sl/python')
	import caffe
	caffe.set_mode_gpu()

	net = caffe.Net(opt.prototxt, opt.caffemodel, caffe.TEST)
	net.blobs['data'].reshape(opt.batch_size, 3, 227, 227)

	return net

def eval_data(net):
	
	train_reps, train_labs = get_reps(train_images, train_labels, net)
	val_reps, val_labs = get_reps(val_images, val_labels, net)
	print 'representations generated'
	
	clf = SVC(verbose=True, probability=True)
	clf.fit(train_reps, train_labs)
	print 'model fit'

	accuracy=clf.score(val_reps, val_labs)
	preds=clf.predict_proba(val_reps)
	AP=average_precision_score(val_labs-1, preds[:,1]) #needs binary, pos class score
	
	print 'Accuracy on PPMI:', accuracy
	print 'AP:', AP


#----------------------------------------------------------------#
train_images, train_labels, val_images, val_labels, img_mean = load_ppmi(opt.data_dir)
opt.img_mean=img_mean


if opt.t7!='nil':
	opt.backend='torch'
	eval_data(torch_net())
elif opt.caffemodel!='nil' and opt.prototxt!='nil': 
	opt.backend='caffe'
	eval_data(caffe_net())	
else:
	print 'Either --t7 or --caffemodel & --prototxt'


raw_input('done') 






