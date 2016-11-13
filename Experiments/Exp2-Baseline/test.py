import sys
import numpy as np
import lmdb
import caffe
from collections import defaultdict
import cPickle as pickle
args = sys.argv
caffe.set_mode_gpu()

#################### METRIC ####### TEMPORARY ####################
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
#############################################################################


# Modify the paths given below
deploy_prototxt_file_path = '../../models/baseline_activity/deploy.prototxt' # Network definition file
caffe_model_file_path = args[1] # Trained Caffe model file
val_lmdb_path_imsitu = '../../project_data/LMDB/imsitu/val_lmdb/'
test_lmdb_path_imsitu = '../../project_data/LMDB/imsitu/test_lmdb/' # Test LMDB database path
test_lmdb_path_imsitu_selection = '../../project_data/LMDB/imsitu_selection/selection_lmdb/'
train_lmdb_path_ppmi = '../../project_data/LMDB/ppmi/train_lmdb/'
test_lmdb_path_ppmi = '../../project_data/LMDB/ppmi/test_lmdb/'
mean_file_binaryproto = '../../caffe-sl/python/caffe/imagenet/ilsvrc_2012_mean.npy' # Mean image file

# Extract mean from the mean image file
mu = np.load(mean_file_binaryproto)
mu = mu.mean(1).mean(1)

# Apply a transformer
# CNN reconstruction and loading the trained weights
net = caffe.Classifier(deploy_prototxt_file_path, caffe_model_file_path,
                       mean=mu,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

def evaluate(net, test_path, ppmi=False, train_path=None):
	lmdb_env = lmdb.open(test_path)
	lmdb_txn = lmdb_env.begin()
	lmdb_cursor = lmdb_txn.cursor()
	label_list = []
	distribution = []
	for key, value in lmdb_cursor:
		image_nm = '_'.join(key.split('_')[1:])
		image_loc = '../../project_data/images/imsitu/' + image_nm
		image = caffe.io.load_image(image_loc)
    		datum = caffe.proto.caffe_pb2.Datum()
        	datum.ParseFromString(value)
        	label = int(datum.label)
		prediction = net.predict([image])
		plabel = prediction[0]
		distribution.append(plabel)
		label_list.append(label)
	
	distribution_mat = np.array(distribution)
	# display metric
	val_banner(label_list, distribution_mat)
	
	return label_list, distribution_mat
	


print 'Evaluating Imsitu validation...'
label, prob_dist = evaluate(net, val_lmdb_path_imsitu)
pickle.dump((label, prob_dist), open('imsitu_validation.p', 'wb'))

print 'Evaluating Imsitu Test ....'
label, prob_dist = evaluate(net, test_lmdb_path_imsitu)
pickle.dump((label, prob_dist), open('imsitu_test.p', 'wb'))

print 'Evaluating Imsitu Selection ...'
label, prob_dist = evaluate(net, test_lmdb_path_imsitu_selection)
pickle.dump((label, prob_dist), open('imsitu_selection.p', 'wb'))

# evaluate over ppmi dataset 
#evaluate(net, test_lmdb_path_ppmi, ppmi=True, train_path=train_lmdb_path_ppmi)
