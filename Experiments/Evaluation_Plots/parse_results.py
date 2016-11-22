import glob
import cPickle as pickle
import numpy as np

def parse_preds(pred_dict, out_fl):
	
	out_fl=out_fl.split('/')[-1].split('.pkl')[0]+'.curve'

	preds=sorted(pred_dict.items(), key=lambda x: x[0])
	files, preds= zip(*preds)
	truth, preds= zip(*preds)
	files, truth, preds= np.array(files), np.array(truth), np.asanyarray(preds)
	num_classes=preds.shape[1]

	# code to generate precision @ k values 
	pred_idx=np.argsort(-preds, 1) #first col is the highest
	prec_at_k=np.zeros(num_classes)
	for idx in range(pred_idx.shape[0]):
		
		for k in range(num_classes):		
			top_k=set(pred_idx[idx,:k+1])
			if truth[idx] in top_k:
				prec_at_k[k:]+=1
				break
		
	prec_at_k/=pred_idx.shape[0]
	
	with open('curves/P@k/'+out_fl, 'w') as f:
		for k in range(prec_at_k.shape[0]):
			f.write('%d\t%f\n'%(k, prec_at_k[k]))


	# code to generate PR curve values

	# set a threshold for counting positives (only for computation)
	# ideally, plot all of them - the low ones will fall at the tail
	# of the curve anyway

	thresh=0.1
	truth_stack=np.vstack([truth]*preds.shape[1]).transpose()
	idx_stack=np.vstack([np.arange(preds.shape[1])]*preds.shape[0])
	
	keep=(preds>thresh)
	
	flat_preds=zip(preds[keep], idx_stack[keep], truth_stack[keep])

	# sort according to the most confident predictions
	sorted_pred=sorted(flat_preds, key=lambda x: -x[0])

	pr_data=[]
	correct, predicted, total=0, 0, preds.shape[0]
	for p_val, p_idx, lab in sorted_pred:
		predicted+=1
		if p_idx==lab:
			correct+=1
		pr_data.append((1.0*correct/predicted, 1.0*correct/total))

	print len(pr_data)

	with open('curves/PR/'+out_fl, 'w') as f:
		for P, R in pr_data:
			f.write('%f\t%f\n'%(P,R))
	


pred_files=glob.glob('../Prediction_Pickles/*.pkl')

for fl in pred_files:
	print fl
	pred_dict=pickle.load(open(fl, 'rb'))
	parse_preds(pred_dict, fl)
