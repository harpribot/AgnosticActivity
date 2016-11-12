import cPickle as pickle
import sys

args = sys.argv
input_imsitu_selections = args[1]
output_imsitu_selections = args[2]
label_file = args[3]
label_map = pickle.load(open(label_file, 'rb'))


def label_creator(input_file, output_file, label_map):
	out = open(output_file, 'wb')
	with open(input_file, 'rb') as names:
		name_lst = names.readlines()
		name_lst = [x.strip('\n') for x in name_lst]
		for image in name_lst:
			activity = image.split('_')[0]
			out.write("%s %s\n" %(image, label_map[activity]))

# test lmdb label .txt creator
label_creator(input_imsitu_selections, output_imsitu_selections, label_map)
