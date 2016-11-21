import h5py
import numpy as np
from util.nn_util import *
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.svm import SVC
import cPickle as pickle


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--load', default='nil')
parser.add_argument('--prefix', default='')
parser.add_argument('--data_dir', default='../../project_data')
parser.add_argument('--model_dir', default='models') #cannot put this outside for now
parser.add_argument('--out_dir', default='../Prediction_Pickles')
opt = parser.parse_args()

#----------------------------------------------------------------#
num_classes=504
banner_line='-'*30
def val_banner(truth, pred):

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



def get_preds(data, net, reps=False):

	files, images, labels = data

	output_preds, output_labels, output_files=[], [], []
	n_val=images.shape[0]
	count, n_batches=0, n_val/opt.batch_size

	for batchFiles, batchInputs, batchLabels in get_batches_in_sequence(data, opt):

		batchLabels, _ = batchLabels # take only activity

		batchPred= net.predict(batchInputs)[1] # 2 is loss
		batchPred= batchPred.asNumpyTensor()
		batchPred= np.exp(batchPred)
		
		if reps:
			batchPred=net.get_reps().asNumpyTensor()

		output_preds.append(batchPred)
		output_labels.append(batchLabels)
		output_files.append(batchFiles)

		if count%50==0:
			print '%s/%s..'%(count, n_batches)
		count+=1

	output_preds=np.vstack(output_preds)[:n_val]
	output_labels=np.hstack(output_labels)[:n_val]-1 # torch <3
	output_files=np.hstack(output_files)[:n_val]
	return output_files, output_preds, output_labels


def torch_net():

	import PyTorchHelpers
	TorchTrainer = PyTorchHelpers.load_lua_class('%s/TorchTrainer.lua'%opt.model_dir, 'TorchTrainer')		
	net = TorchTrainer(vars(opt))
	return net

def eval_data(net, data, out_file):
	
	pred_data = get_preds(data, net)

	pred_dict={fl: (lab, pred) for (fl, pred, lab) in zip(*pred_data)}
	if len(out_file)>0:	
		pickle.dump(pred_dict, open('%s/%s'%(opt.out_dir, out_file),'wb'))

	val_files, val_pred, val_labs= pred_data
	val_banner(val_labs, val_pred)

def eval_ppmi(net, data, out_file):
	
	train_data, val_data = data

	_, train_reps, train_labels = get_preds(train_data, net, True)
	val_files, val_reps, val_labels = get_preds(val_data, net, True)
	
	print 'Training SVM...'
	clf = SVC(verbose=False, probability=True)
	clf.fit(train_reps, train_labels)

	accuracy=clf.score(val_reps, val_labels)
	val_preds=clf.predict_proba(val_reps)
	print 'SVM predictions generated'

	AP=average_precision_score(val_labels, val_preds[:,1]) #needs binary, pos class score
	
	print 'Accuracy on PPMI:', accuracy
	print 'AP:', AP

	pred_dict={fl: (lab, pred) for (fl, pred, lab) in zip(val_files, val_preds, val_labels)}
	if len(out_file)>0:	
		pickle.dump(pred_dict, open('%s/%s'%(opt.out_dir, out_file),'wb'))

#-------------------------------------------------------#

net=torch_net()
opt.pad=1

#----------------------- imSitu ------------------------#
_, val_data, test_data, img_mean = load_imsitu('%s/hdf5/imsitu/'%opt.data_dir)
opt.img_mean=img_mean

print 'evaluating imSitu Val'
eval_data(net, val_data, '%s_imsitu_val.pkl'%opt.prefix)
print 'evaluating imSitu Test'
eval_data(net, test_data, '%s_imsitu_test.pkl'%opt.prefix)

#-------------------- imSitu Subset --------------------#
val_data, img_mean = load_subset('%s/hdf5/imsitu_subset/'%opt.data_dir)
opt.img_mean=img_mean

print 'evaluating imSitu subset'
eval_data(net, val_data, '%s_subset.pkl'%opt.prefix)

##------------------------- PPMI ------------------------#
#train_data, val_data, img_mean = load_ppmi('%s/hdf5/ppmi/'%opt.data_dir)
#opt.img_mean=img_mean

#print 'evaluating PPMI'
#eval_ppmi(net, [train_data, val_data], '%s_ppmi.pkl'%opt.prefix)

##-------------------------------------------------------#

#raw_input('done') 






