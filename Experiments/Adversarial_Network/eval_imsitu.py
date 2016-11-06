import h5py
import numpy as np
from util.nn_util import *
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#don't worry, we'll put it back in there
#model_dir='../../models/adversarial_object_activity'
model_dir='models'
data_dir='../../project_data'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--data_dir', default='%s/hdf5/imsitu'%data_dir)
parser.add_argument('--t7', default='nil')
parser.add_argument('--prototxt', default='nil')
parser.add_argument('--caffemodel', default='nil')
parser.add_argument('--final_layer', default='fc8') #'fc8_p'/'fc8-new'
opt = parser.parse_args()

#----------------------------------------------------------------#
banner_line='-'*30
def val_banner(truth, pred):

	truth=truth-1 # <3 lua

	pred_idx=np.argsort(pred, 1)
	acc1, acc5=0.0, 0.0
	for idx in range(pred_idx.shape[0]):
		top5=set(pred_idx[idx,-5:])
		if truth[idx] in top5:
			acc5+=1
		if truth[idx]==pred_idx[idx,-1]:
			acc1+=1
	acc1, acc5=acc1/len(truth), acc5/len(truth)

	MAP=1

	val_banner='p@1=%.3f | p@5=%.3f | MAP=%.3f'\
			%(acc1, acc5, MAP)
	val_banner='%s\n%s\n%s'%(banner_line, val_banner, banner_line)
	print val_banner

	return [acc1, acc5, MAP]

def get_preds(inputs, labels, net):

	output_preds, output_labels=[], []
	count, n_batches=0, inputs.shape[0]/opt.batch_size

	for batchInputs, batchLabels in get_batches_in_sequence(inputs, labels, opt):

		batchLabels, _ = batchLabels # take only activity

		if opt.backend=='torch':
			batchPred= net.predict(batchInputs)[1] # 2 is loss
			batchPred= batchPred.asNumpyTensor()
		elif opt.backend=='caffe':
			net.blobs['data'].data[...] = batchInputs[:,::-1,:,:]
			net.forward()
			batchPred = net.blobs[opt.final_layer].data.copy()

		output_preds.append(batchPred)
		output_labels.append(batchLabels)

		if count%10==0:
			print '%s/%s..'%(count, n_batches)
		count+=1

	output_preds=np.vstack(output_preds)
	output_labels=np.hstack(output_labels)	
	return output_preds, output_labels 


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
	
	val_pred, val_labs = get_preds(val_images, val_labels, net)
	print 'predictions generated'
	
	val_banner(val_labs, val_pred)


#----------------------------------------------------------------#
train_images, train_labels, val_images, val_labels, img_mean = load_data(opt.data_dir)
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






