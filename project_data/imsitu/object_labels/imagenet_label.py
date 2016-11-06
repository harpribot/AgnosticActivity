import re
import collections
import cPickle as pickle

with open('imagenet1000.txt','r') as f:
	imagenet_labels=f.read().strip().split('\n')

label_dict, ilabel_dict={}, {}
for idx, line in enumerate(imagenet_labels):
	objs=line.split(', ')
	for o in objs:
		label_dict[o.lower().strip()]=set([idx])
	ilabel_dict[idx]=line	
	

#add the generalizations too
for obj in label_dict.keys():
	parts=obj.split(' ')
	tail=parts[-1].strip()	
	if len(parts)==1 or len(tail)<=1:
		continue
	
	if tail not in label_dict:
		label_dict[tail]=set()

	label_dict[tail]|=label_dict[obj] 

'''
#add extra custom links
with open('custom_links.txt','r') as f:
	lines=f.read().strip().split('\n')
	lines=[l.split(':') for l in line]
	for obj, new_obj in lines:
		label_dict[new_obj]=label_dict[obj]
'''

with open('OpenFrame500.tab','r') as f:
	imsitu_labels=f.read().strip().split('\n')

search_pat=re.compile('\(([^;]+);[^;]+;\[([^\]]+)\]')
slot_dict=collections.defaultdict(set)

imsitu_dict={}
for line in imsitu_labels:
	fl, obj_str=line.split('\t')
	imsitu_dict[fl]=set()

	line_objs=set()
	for slot, obj_string in search_pat.findall(obj_str.lower()):

		#if slot in ['place','agent']:
		#	continue

		objs= [o.strip() for o in obj_string.split(', ')]
		slot_dict[slot]|=set(objs)
		for o in objs:
			if o in label_dict:
				imsitu_dict[fl]|=label_dict[o]
		line_objs|=set(objs)

		#if len(imsitu_dict[fl])==0:
			#print line
			 

#avg_labels=[1.0*len(imsitu_dict[fl]) for fl in imsitu_dict if len(imsitu_dict[fl])>0]
coverage=1.0*sum([1 for fl in imsitu_dict if len(imsitu_dict[fl])>0])/len(imsitu_dict)
print coverage

pickle.dump(imsitu_dict, open('obj_labels.pkl','wb'))


