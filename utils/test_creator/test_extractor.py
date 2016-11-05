import cPickle as pickle
import sys
import numpy as np
import operator

arg = sys.argv
image_obj_dict = pickle.load(open(arg[1], 'rb'))
mean_threshold_multiplier = int(arg[2])
least_k = int(arg[3])

# Do statistical analysis to find the most famous object for each of the activity
activity_obj_cnt = dict()
obj_activity_cnt = dict()
activity_images_lst = dict()
for key, objects in image_obj_dict.iteritems():
    activity = key.split('_')[0]
    if activity not in activity_images_lst:
        activity_images_lst[activity] = [key]
    else:
        activity_images_lst[activity].append(key)

    if activity not in activity_obj_cnt:
        activity_obj_cnt[activity] = dict()
    else:
        for obj in objects:
            if obj not in activity_obj_cnt[activity]:
                activity_obj_cnt[activity][obj] = 1
            else:
                activity_obj_cnt[activity][obj] += 1

    for obj in objects:
        if obj not in obj_activity_cnt:
            obj_activity_cnt[obj] = dict()
        else:
            if activity not in obj_activity_cnt[obj]:
                obj_activity_cnt[obj][activity] = 1
            else:
                obj_activity_cnt[obj][activity] += 1

pickle.dump(activity_obj_cnt, open('activity_obj_cnt.p', 'wb'))
pickle.dump(obj_activity_cnt, open('obj_activity_cnt.p', 'wb'))

# Now basically find a list of all objects that are 3x above the mean count
# of all objects for that activity
activity_bias_obj_pair = dict()
for activity, obj_count in activity_obj_cnt.iteritems():
    mean = np.mean(np.array(obj_count.values()))
    threshold = mean_threshold_multiplier * mean

    biased_objects = [x for x in obj_count.keys() if obj_count[x] >= threshold]
    activity_bias_obj_pair[activity] = biased_objects

pickle.dump(activity_bias_obj_pair, open('activity_bias_obj_pair.p', 'wb'))

# Now take all these objects and then find the activities in which it is least present
# Then take all the images for these activites if it is present in those images
test_set = set()
for activity, biased_objects in activity_bias_obj_pair.iteritems():
    for obj in biased_objects:
        activities_and_cnt = obj_activity_cnt[obj]
        sorted_activities =sorted(activities_and_cnt.items(), key=operator.itemgetter(1))
        least_frequent_activities = [x[0] for x in sorted_activities[:least_k]]

        for bottom_activity in least_frequent_activities:
            if bottom_activity != activity:
                for image in activity_images_lst[bottom_activity]:
                    if obj in image_obj_dict[image]:
                        test_set.add(image)

pickle.dump(list(test_set), open('test_images.p', 'wb'))
