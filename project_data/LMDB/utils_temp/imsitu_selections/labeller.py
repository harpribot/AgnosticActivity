input_file = 'dev.txt'
output_file = 'dev_label.txt'
out = open(output_file, 'wb')
counter = 0
label_map = dict()
with open(input_file) as names:
	name_lst = names.readlines()
	name_lst = [x.strip('\n') for x in name_lst]
	for image in name_lst:
		activity = image.split('_')[0]
		if activity not in label_map:
			label_map[activity] = counter
			counter += 1
		
		out.write("%s %s\n" %(image, label_map[activity]))

# sanity check
print counter
assert counter == 504, 'Labels not equal to total number of possible activities'
out.close()
	
