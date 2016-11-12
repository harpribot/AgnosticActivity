import cPickle as pickle
import sys

args = sys.argv
input_file_train = args[1]
input_file_dev = args[2]
input_file_test = args[3]
output_file_train = args[4]
output_file_dev = args[5]
output_file_test = args[6]
label_file = args[7]
label_map = pickle.load(open(label_file, 'rb'))


def label_creator(input_file, output_file, label_map):
	out = open(output_file, 'wb')
	with open(input_file, 'rb') as names:
		name_lst = names.readlines()
		name_lst = [x.strip('\n') for x in name_lst]
		for image in name_lst:
			activity = image.split('_')[0]
			out.write("%s %s\n" %(image, label_map[activity]))
	out.close()

# train lmdb label .txt creator
print input_file_train, output_file_train
label_creator(input_file_train, output_file_train, label_map)
# dev lmdb label .txt creator
print input_file_dev, output_file_dev
label_creator(input_file_dev, output_file_dev, label_map)
# test lmdb label .txt creator
print input_file_test, output_file_test
label_creator(input_file_test, output_file_test, label_map)
