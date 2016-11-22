import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import glob
import sys
import numpy as np


def plotPR(curve_files, out_file):
	for curve_fl in curve_files:
		with open(curve_fl,'r') as f:
			data=f.read().strip().split('\n')
		data=[map(float, d.split('\t')) for d in data]
		prec, rec=zip(*data)

		curve_name=curve_fl.split('/')[-1]
		plt.plot(rec, prec, label=curve_name)

	plt.xlabel('recall')
	plt.ylabel('precision')

	plt.legend()
	plt.savefig(out_file)
	plt.clf()

def plotP_at_K(curve_files, out_file):
	for curve_fl in curve_files:
		with open(curve_fl,'r') as f:
			data=f.read().strip().split('\n')
		data=[map(float, d.split('\t')) for d in data]
		data=np.array(data)

		curve_name=curve_fl.split('/')[-1]
		plt.plot(data[:,0], data[:,1], label=curve_name)	

	plt.xlabel('K')
	plt.ylabel('precision@K')

	plt.legend()
	plt.savefig(out_file)
	plt.clf()


plotP_at_K(glob.glob('curves/P@k/*subset.curve'), 'P@K_subset.png')
plotP_at_K(glob.glob('curves/P@k/*val.curve'), 'P@K_imval.png')
plotP_at_K(glob.glob('curves/P@k/*test.curve'), 'P@K_imtest.png')
	
plotPR(glob.glob('curves/PR/*subset.curve'), 'PR_subset.png')
plotPR(glob.glob('curves/PR/*val.curve'), 'PR_imval.png')
plotPR(glob.glob('curves/PR/*test.curve'), 'PR_imtest.png')

