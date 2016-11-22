import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import glob
import sys
import numpy as np


for curve_fl in sys.argv[1:]:
	with open(curve_fl,'r') as f:
		data=f.read().strip().split('\n')
	data=[map(float, d.split('\t')) for d in data]
	data=np.array(data)

	curve_name=curve_fl.split('/')[-1]
	plt.plot(data[:,0], data[:,1], label=curve_name)


plt.legend()
plt.savefig('P@k.png')
