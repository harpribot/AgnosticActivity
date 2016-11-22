import glob
import cPickle as pickle
import numpy as np

def parse_preds(pred_dict, out_fl):
	
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
	
	out_fl=out_fl.split('/')[-1].split('.pkl')[0]+'.curve'
	out_fl='curves/P@k/'+out_fl

	with open(out_fl, 'w') as f:
		for k in range(prec_at_k.shape[0]):
			f.write('%d\t%f\n'%(k, prec_at_k[k]))


	'''
	# code to generate PR curve values
	
	sorted_pred=sorted(zip(preds, truth), key=lambda x: -x[0])


	pr_data=[]
	correct, predicted=0, 0
	for b_no, p_idx, p_val in sorted_pred:
		predicted+=1
		if p_idx in block_gold[b_no]:
			correct+=1
		pr_data.append((1.0*correct/predicted, 1.0*correct/total))
	'''



pred_files=glob.glob('../Prediction_Pickles/*.pkl')

for fl in pred_files:
	print fl
	pred_dict=pickle.load(open(fl, 'rb'))
	parse_preds(pred_dict, fl)
