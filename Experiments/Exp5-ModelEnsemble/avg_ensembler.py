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
corrupt_images = ['decomposing_205.jpg',
'fueling_67.jpg',
'imitating_110.jpg',
'knocking_90.jpg',
'lifting_267.jpg',
'peeing_138.jpg',
'tickling_209.jpg',
'tripping_245.jpg']

def evaluate_ensemble(results_mdl_a, results_mdl_b):
	results_a = pickle.load(open(results_mdl_a, 'rb'))
	results_b = pickle.load(open(results_mdl_b, 'rb'))
	dist_mdl_a = []
	dist_mdl_b = []
	true_label = []
	
	for key, value in results_a.iteritems():
		label_a, dist_a = value
		if key not in corrupt_images:
			true_label.append(label_a)
                	dist_mdl_a.append(dist_a)
			label_b, dist_b = results_b[key]
			assert label_a == label_b, 'The labels are not the same'
			dist_mdl_b.append(dist_b)

	dist_mdl_a = np.array(dist_mdl_a)
	dist_mdl_b = np.array(dist_mdl_b) 
	ensemble_dist = (dist_mdl_a + dist_mdl_b)/2.0
	true_label = np.array(true_label)
	val_banner(true_label, ensemble_dist)


print 'Loading the baseline and model1 distribution'
model1_imsitu_val = '../Exp3-Siamese_Model1/imsitu_validation.p'
model1_imsitu_test = '../Exp3-Siamese_Model1/imsitu_test.p'
model1_imsitu_selection = '../Exp3-Siamese_Model1/imsitu_selection.p'

baseline_imsitu_val = '../Exp2-Baseline/imsitu_validation.p'
baseline_imsitu_test = '../Exp2-Baseline/imsitu_test.p'
baseline_imsitu_selection = '../Exp2-Baseline/imsitu_selection.p'

model2_imsitu_val = '../Exp4-Adversarial_Model2/imsitu_validation.p'
model2_imsitu_selection = '../Exp4-Adversarial_Model2/imsitu_selection.p'

baseline_tushar_val = '../baseline_tushar_validation.p'
baseline_tushar_selection = '../baseline_tushar_selection.p'

print 'Evaluatin for Baseline and Model1...'
print '.....'
print 'Evaluating Ensemble for Imsitu Validation'
evaluate_ensemble(baseline_imsitu_val, model1_imsitu_val)
print 'Evaluating Ensemlbe for Imsitu Test'
evaluate_ensemble(baseline_imsitu_test, model1_imsitu_test)
print 'Evaluating Ensemble for Imsitu Selection'
evaluate_ensemble(baseline_imsitu_selection, model1_imsitu_selection)


print 'Evaluatin for Model1 and Model2...'
print '.....'
print 'Evaluating Ensemble for Imsitu Validation'
evaluate_ensemble(model1_imsitu_val, model2_imsitu_val)
#print 'Evaluating Ensemlbe for Imsitu Test'
#evaluate_ensemble(baseline_imsitu_test, model1_imsitu_test)
print 'Evaluating Ensemble for Imsitu Selection'
evaluate_ensemble(model1_imsitu_selection, model2_imsitu_selection)


print 'Evaluatin for Basleline Tushar and Model1...'
print '.....'
print 'Evaluating Ensemble for Imsitu Validation'
evaluate_ensemble(baseline_tushar_val, model1_imsitu_val)
#print 'Evaluating Ensemlbe for Imsitu Test'
#evaluate_ensemble(baseline_imsitu_test, model1_imsitu_test)
print 'Evaluating Ensemble for Imsitu Selection'
evaluate_ensemble(baseline_tushar_selection, model1_imsitu_selection)


print 'Evaluatin for Baseline Tushar and Model2...'
print '.....'
print 'Evaluating Ensemble for Imsitu Validation'
evaluate_ensemble(baseline_tushar_val, model2_imsitu_val)
#print 'Evaluating Ensemlbe for Imsitu Test'
#evaluate_ensemble(baseline_imsitu_test, model1_imsitu_test)
print 'Evaluating Ensemble for Imsitu Selection'
evaluate_ensemble(baseline_tushar_selection, model2_imsitu_selection)


print 'Evaluatin for Baseline and Model2...'
print '.....'
print 'Evaluating Ensemble for Imsitu Validation'
evaluate_ensemble(baseline_imsitu_val, model2_imsitu_val)
#print 'Evaluating Ensemlbe for Imsitu Test'
#evaluate_ensemble(baseline_imsitu_test, model1_imsitu_test)
print 'Evaluating Ensemble for Imsitu Selection'
evaluate_ensemble(baseline_imsitu_selection, model2_imsitu_selection)


