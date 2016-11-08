import cPickle as pickle
import collections
import math

#{filename: set([obj1, obj2...])}
file_obj_map=pickle.load(open('obj_labels_no_grounding.pkl','rb'))

# remove all files that are in train
with open('train.txt', 'r') as f:
	train_files=f.read().strip().split('\n')
	for fl in train_files:
		if fl in file_obj_map:
			file_obj_map.pop(fl)

# retrieve the set of all objects present in the files
obj_set=set.union(*[file_obj_map[fl] for fl in file_obj_map])

#{action: [fl1, fl2...]}
act_fl_map=collections.defaultdict(list)
for fl in file_obj_map:
	act_fl_map[fl.split('_')[0]].append(fl)
activity_list=sorted(act_fl_map.keys())

'''
#{action: {obj1:1.0, obj2:0.8, obj3:0.01}
tf={}
for idx, act in enumerate(activity_list):
	act_files=act_fl_map[act]
	tf_dict=[]
	for obj in obj_set:
		file_count=len([fl for fl in act_files if obj in file_obj_map[fl]])
		tf_dict.append((obj, 1.0*file_count/len(act_files)))
	tf[act]=dict(tf_dict)
	if idx%10==0:
		print '%s/%s..'%(idx, len(activity_list))

print 'tf constructed'

#{obj: 0.8} -- sum of scores across all activities
df={}
for obj in obj_set:
	df[obj]=1.0*sum([tf[act][obj] for act in tf])
print 'df constructed'

pickle.dump([tf, df], open('tf_df_dicts.pkl','wb'))
#----------------------------------------------------------------#
'''

tf, df = pickle.load(open('tf_df_dicts.pkl','rb'))
agnostic_set=set()

for idx, act in enumerate(activity_list):

	tfidf=[]
	for obj in tf[act]:
		if tf[act][obj]==df[obj]: #word appears ONLY in this activity
			continue
		tfidf.append((obj, tf[act][obj]/(df[obj]-tf[act][obj])))

	tfidf=sorted(tfidf, key=lambda x: -x[1])

	#look at top word
	for obj, score in tfidf[0:10]:
		act_files, other_files=[], []
		for fl in file_obj_map:
			fl_act=fl.split('_')[0]		
			if obj in file_obj_map[fl]:
				if fl_act==act:
					act_files.append(fl)
				else:
					other_files.append(fl)

		other_acts=set([fl.split('_')[0] for fl in other_files])
		n_act, n_other=len(act_files), len(other_files)

		if n_act>20 and 4*n_other<=n_act and n_other>1:
			#print act, obj, score
			#print n_act, n_other, len(other_acts)
			agnostic_set|=set(other_files)

	print '%s/%s.. %s'%(idx, len(activity_list), len(agnostic_set))
	#raw_input('-'*30)

with open('imsitu_subset.txt','w') as f:
	f.write('\n'.join(list(agnostic_set)))


