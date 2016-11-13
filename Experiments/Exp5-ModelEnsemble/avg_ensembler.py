import numpy as np
import cPickle as pickle

######################## Metric TEMPORARY #############################3
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

#####################################################################
def evaluate_ensemble(baseline_pickle, model1_pickle):
	label_list, dist_baseline = pickle.load(open(baseline_pickle, 'rb'))
	_, dist_model1 = pickle.load(open(model1_pickle, 'rb'))
	print dist_baseline.shape
	print dist_model1.shape
	ensemble_dist = (dist_baseline + dist_model1)/2.0

	val_banner(label_list, ensemble_dist)


print 'Loading the baseline and model1 distribution'
model1_imsitu_val = '../Exp3-Siamese_Model1/imsitu_validation.p'
model1_imsitu_test = '../Exp3-Siamese_Model1/imsitu_test.p'
model1_imsitu_selection = '../Exp3-Siamese_Model1/imsitu_selection.p'

baseline_imsitu_val = '../Exp2-Baseline/imsitu_validation.p'
baseline_imsitu_test = '../Exp2-Baseline/imsitu_test.p'
baseline_imsitu_selection = '../Exp2-Baseline/imsitu_selection.p'

print 'Evaluating Ensemble for Imsitu Validation'
evaluate_ensemble(baseline_imsitu_val, model1_imsitu_val)

print 'Evaluating Ensemlbe for Imsitu Test'
evaluate_ensemble(baseline_imsitu_test, model1_imsitu_test)

print 'Evaluating Ensemble for Imsitu Selection'
evaluate_ensemble(baseline_imsitu_selection, model1_imsitu_selection)
	
