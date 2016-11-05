import caffe

net = caffe.Net('models/dissimilarity_siamese_net/deploy_dissimilarity.prototxt', 'models/dissimilarity_siamese_net/pretrained_model/bvlc_alexnet.caffemodel', caffe.TEST)

params = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7','conv1_p', 'conv2_p', 'conv3_p', 'conv4_p', 'conv5_p', 'fc6_p', 'fc7_p']
old_params = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
new_params = ['conv1_p', 'conv2_p', 'conv3_p', 'conv4_p', 'conv5_p', 'fc6_p', 'fc7_p']

params_val = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for old, new in zip(old_params, new_params):
    params_val[new][0].flat = params_val[old][0].flat
    params_val[new][1][...] = params_val[old][1]

net.save('models/dissimilarity_siamese_net/bvlc_dissimilarity.caffemodel')
