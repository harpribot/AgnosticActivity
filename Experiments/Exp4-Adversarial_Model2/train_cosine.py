import PyTorchHelpers
import h5py
import numpy as np
from util.nn_util import *
from sklearn.metrics import accuracy_score

#don't worry, we'll put it back in there
#model_dir='../../models/adversarial_object_activity'
model_dir='models'
data_dir='../../project_data'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_acts', type=int, default=504)
parser.add_argument('--data_dir', default='%s/hdf5/imsitu'%data_dir)
parser.add_argument('--eval_val_every', type=int, default=1)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--cv_dir', default='cv/tmp/')
parser.add_argument('--load', default='nil')
parser.add_argument('--log', default='')
parser.add_argument('--model', default='%s/cosine'%model_dir)
parser.add_argument('--grl_lambda', type=float, default=1.0)
#optimization
parser.add_argument('--base_wd', type=float, default=1e-4)
parser.add_argument('--base_lr', type=float, default=0.01)
opt = parser.parse_args()

batch_size=opt.batch_size
if len(opt.log)>0:
	delete_file(opt.log+'.train')
	delete_file(opt.log+'.val')

#----------------------------------------------------------------#

def unpack(table):
	return [table[idx] for idx in range(1, len(table.keys())+1)]

def log_data(data, trainval):
	if len(opt.log)==0:
		return

	if trainval=='train':
		epoch, batch_frac, loss_stats= data
		loss, nll_loss, cos_loss= loss_stats
		with open(opt.log+'.train','a') as f:
			f.write('%d,%f,%f,%f,%f\n'%(epoch, batch_frac, loss, nll_loss, cos_loss))
		return

	if trainval=='val':
		epoch, loss, val_stats= data
		acc1, acc5, MAP, MRR= val_stats

		write_out='%d,%f,%f,%f,%f,%f\n'%(epoch, loss, acc1, acc5, MAP, MRR)
		with open(opt.log+'.val','a') as f:
			f.write(write_out)
		return



banner_line='-'*30
def val_banner(truth, pred, epoch, val_loss):

	truth=truth-1 # <3 lua

	pred_idx=np.argsort(-pred, 1) #first col is the highest
	prec_at_k=np.zeros(opt.num_acts)
	MRR=0.0

	for idx in range(pred_idx.shape[0]):
		
		rank=pred_idx[idx].tolist().index(truth[idx])+1
		MRR+=1.0/rank

		for k in range(opt.num_acts):		
			top_k=set(pred_idx[idx,:k+1])
			if truth[idx] in top_k:
				prec_at_k[k:]+=1
				break
	prec_at_k/=pred_idx.shape[0]
	MAP=np.mean(prec_at_k)
	MRR/=pred_idx.shape[0]

	acc1, acc5=prec_at_k[0], prec_at_k[4]

	val_banner='epoch: %d - loss: %.3f | p@1=%.3f | p@5=%.3f | MAP=%.3f | MRR=%.3f'\
			%(epoch, val_loss, acc1, acc5, MAP, MRR)
	val_banner='%s\n%s\n%s'%(banner_line, val_banner, banner_line)
	print val_banner

	return [acc1, acc5, MAP, MRR]



def epochToLearningRate(epoch):
	if epoch < 100:
		return opt.base_lr
	if epoch < 200:
		return opt.base_lr/10.0
	return opt.base_lr/100.0


#----------------------------------------------------------------#
train_data, val_data, _, img_mean= load_imsitu(opt.data_dir)
_, train_images, train_labels= train_data
_, val_images, val_labels= val_data
opt.img_mean=img_mean

TorchTrainer = PyTorchHelpers.load_lua_class('%s/TorchTrainer.lua'%model_dir, 'TorchTrainer')
net = TorchTrainer(vars(opt))

n_train, n_val= train_images.shape[0], val_images.shape[0]
n_train_batches= n_train/batch_size
n_val_batches= n_val/batch_size
epoch = net.epoch

# +objs
opt.pad=1

while True:

	if epoch%opt.eval_val_every==0 and epoch>net.epoch: 

		val_truth, val_pred=[], []
		val_loss=0.0
		count=0
		for _, batchInputs, batchLabels in get_batches_in_sequence(val_data, opt):

			batchPred = net.predict(batchInputs, batchLabels)
			batchPred, loss= batchPred[1], batchPred[2]
			val_loss+=loss

			val_pred.append(batchPred.asNumpyTensor())
			val_truth.append(batchLabels[0])
			if count%10==0:
				print '%s/%s..'%(count, n_val_batches)

			count+=1
	
		val_pred, val_truth= np.vstack(val_pred)[:n_val], np.hstack(val_truth)[:n_val]
		val_loss/=n_val_batches

		val_stats=val_banner(val_truth, val_pred, epoch, val_loss)

		# log everything we have
		log_data([epoch-1, val_loss, val_stats], 'val')


		if epoch%opt.save_every==0:
			net.save(epoch, val_loss)
			print 'model saved'


	learningRate = opt.base_lr #epochToLearningRate(epoch)
	epochLoss = 0
	for b in range(n_train_batches):

		batchInputs, batchLabels=get_random_batch(train_data, opt)
		loss_stats= net.trainBatch(batchInputs, batchLabels, learningRate)
		loss, nll_loss, cos_loss=unpack(loss_stats)


		if b%opt.print_every==0:
			print('  epoch: %d | batch: %d/%d | loss: %.6f | NLL: %.3f | COS: %.3e' %(epoch, b, n_train_batches, loss, nll_loss, cos_loss))
			log_data([epoch-1, 1.0*b/n_train_batches, [loss, nll_loss, cos_loss]], 'train')

		epochLoss += loss

	epoch += 1


