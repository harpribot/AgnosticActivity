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
parser.add_argument('--model', default='%s/adversary'%model_dir)
parser.add_argument('--grl_lambda', type=float, default=0.5)
#optimization
parser.add_argument('--base_wd', type=float, default=1e-4)
parser.add_argument('--base_lr', type=float, default=0.01)
opt = parser.parse_args()

batch_size=opt.batch_size
if len(opt.log)>0:
	delete_file(opt.log+'.train')
	delete_file(opt.log+'.val')

#----------------------------------------------------------------#


def log_data(data, trainval):
	if len(opt.log)==0:
		return

	if trainval=='train':
		epoch, batch_frac, loss= data
		with open(opt.log+'.train','a') as f:
			f.write('%d,%f,%f\n'%(epoch, batch_frac, loss))
		return

	if trainval=='val':
		epoch, loss, val_stats= data
		acc1, acc5= val_stats

		write_out='%d,%f,%f,%f\n'%(epoch, loss, acc1, acc5)
		with open(opt.log+'.val','a') as f:
			f.write(write_out)
		return


banner_line='-'*30
def val_banner(truth, pred, epoch, loss):

	truth=truth-1 # <3 lua
	pred=np.argsort(pred, 1)
	acc1, acc5=0.0, 0.0
	for idx in range(pred.shape[0]):
		top5=set(pred[idx,-5:])
		if truth[idx] in top5:
			acc5+=1
		if truth[idx]==pred[idx,-1]:
			acc1+=1
	acc1, acc5=acc1/len(truth), acc5/len(truth)

	val_banner='epoch: %d | loss=%.6f | p@1=%.3f | p@5=%.3f'\
			%(epoch, loss, acc1, acc5)
	val_banner='%s\n%s\n%s'%(banner_line, val_banner, banner_line)
	print val_banner

	return [acc1, acc5]


def epochToLearningRate(epoch):
   if epoch < 18:
	  return opt.base_lr
   if epoch < 50:
	  return opt.base_lr/10.0
   return opt.base_lr/100.0


#----------------------------------------------------------------#
train_images, train_labels, val_images, val_labels, img_mean = load_data(opt.data_dir)
opt.img_mean=img_mean

TorchTrainer = PyTorchHelpers.load_lua_class('%s/AdversaryTrainer.lua'%model_dir, 'AdversaryTrainer')
net = TorchTrainer(vars(opt))

n_train, n_val= train_images.shape[0], val_images.shape[0]
n_train_batches= n_train/batch_size
n_val_batches= n_val/batch_size
epoch = net.epoch

# +objs
net.reset_lr_mults()
net.set_grl_lambda(opt.grl_lambda)


while True:

	if epoch%opt.eval_val_every==0: 

		val_truth, val_pred=[], []
		val_loss=0.0
		count=0
		for batchInputs, batchLabels in get_batches_in_sequence(val_images, val_labels, opt):

			batchPred = net.predict(batchInputs, batchLabels)
			batchPred, loss= batchPred[1], batchPred[2]
			val_loss+=loss

			val_pred.append(batchPred.asNumpyTensor())
			val_truth.append(batchLabels[0])
			if count%10==0:
				print '%s/%s..'%(count, n_val_batches)

			count+=1
	
		val_pred, val_truth= np.vstack(val_pred), np.hstack(val_truth)
		val_loss/=n_val_batches

		val_stats=val_banner(val_truth, val_pred, epoch, val_loss)

		# log everything we have
		log_data([epoch-1, val_loss, val_stats], 'val')


		if epoch%opt.save_every==0:
			net.save(epoch, val_loss)
			print 'model saved'


	learningRate = epochToLearningRate(epoch)
	epochLoss = 0
	for b in range(n_train_batches):

		batchInputs, batchLabels=get_random_batch(train_images, train_labels, opt)
		loss = net.trainBatch(batchInputs, batchLabels, learningRate)
		if b%opt.print_every==0:
			print('  epoch: %d | batch: %d/%d | loss: %.6f' %(epoch, b, n_train_batches, loss))
			log_data([epoch-1, 1.0*b/n_train_batches, loss], 'train')

		epochLoss += loss

	epoch += 1


