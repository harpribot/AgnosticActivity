import h5py
import numpy as np
from Adversarial_Network.util.nn_util import *
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import cPickle as pickle

#don't worry, we'll put it back in there
#model_dir='../../models/adversarial_object_activity'
model_dir='Adversarial_Network/models'
data_dir='../project_data'

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
num_classes=504
banner_line='-'*30
def val_banner(truth, pred):

	truth=truth-1 # <3 lua

	pred_idx=np.argsort(-pred, 1) #first col is the highest
	prec_at_k=np.zeros(num_classes)
	MRR=0.0

	for idx in range(pred_idx.shape[0]):
		
		rank=pred_idx[idx].tolist().index(truth[idx])+1
		MRR+=1.0/rank

		for k in range(num_classes):		
			top_k=set(pred_idx[idx,:k+1])
			if truth[idx] in top_k:
				prec_at_k[k:]+=1
				break
	prec_at_k/=pred_idx.shape[0]
	MAP=np.mean(prec_at_k)
	MRR/=pred_idx.shape[0]

	acc1, acc5=prec_at_k[0], prec_at_k[4]

	val_banner='p@1=%.3f | p@5=%.3f | MAP=%.3f | MRR=%.3f'\
			%(acc1, acc5, MAP, MRR)
	val_banner='%s\n%s\n%s'%(banner_line, val_banner, banner_line)
	print val_banner

def get_preds(data, net):

	files, images, labels = data

	output_preds, output_labels, output_files=[], [], []
	n_val=images.shape[0]
	count, n_batches=0, n_val/opt.batch_size

	for batchFiles, batchInputs, batchLabels in get_batches_in_sequence(data, opt):

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
		output_files.append(batchFiles)

		if count%10==0:
			print '%s/%s..'%(count, n_batches)
		count+=1

	output_preds=np.vstack(output_preds)[:n_val]
	output_labels=np.hstack(output_labels)[:n_val]
	output_files=np.hstack(output_files)[:n_val]
	return output_files, output_preds, output_labels


def torch_net():

	import PyTorchHelpers
	opt.load=opt.t7 #so that TorchTrainer can load
	#TorchTrainer = PyTorchHelpers.load_lua_class('%s/AdversaryTrainer.lua'%model_dir, 'AdversaryTrainer')
	TorchTrainer = PyTorchHelpers.load_lua_class('%s/CosineTrainer.lua'%model_dir, 'CosineTrainer')		
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
	
	val_files, val_pred, val_labs = get_preds(val_data, net)
	print 'predictions generated'
	
	print val_pred.shape

	val_banner(val_labs, val_pred)


#----------------------------------------------------------------#
_, val_data, img_mean = load_imsitu(opt.data_dir)
opt.img_mean=img_mean
opt.eval=1 #to get filenames as well

if opt.t7!='nil':
	opt.backend='torch'
	eval_data(torch_net())
elif opt.caffemodel!='nil' and opt.prototxt!='nil': 
	opt.backend='caffe'
	eval_data(caffe_net())	
else:
	print 'Either --t7 or --caffemodel & --prototxt'


raw_input('done') 






