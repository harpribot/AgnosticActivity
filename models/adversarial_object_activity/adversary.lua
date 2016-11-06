require 'cudnn'
require 'nngraph'
require 'nnlr'
require 'paths'

function load_alexnet()

	local caffe_t7='models/pretrained/bvlc_reference_caffenet.t7'
	if paths.filep(caffe_t7) then
		local model= torch.load(caffe_t7)
		return model
	end

	require 'loadcaffe'
	local pretrain_dir='models/pretrained/'
	local deploy= deploy or pretrain_dir..'deploy.prototxt'
	local weights=weights or pretrain_dir..'bvlc_reference_caffenet.caffemodel'

	local model = loadcaffe.load(deploy, weights, 'cudnn')
	torch.save(caffe_t7, model)
	print 'bvlc model saved as .t7'

	return model
end

function set_lr(module, factor)
	if module.weight~=nil then
		module:learningRate('weight', factor)
			:learningRate('bias', factor)
			:weightDecay('weight', factor)
			:weightDecay('bias', factor)
	end
end

function build_model(params) --requires (b,3,227,227) input

	local SpatialConvolution = cudnn.SpatialConvolution
	local SpatialMaxPooling = cudnn.SpatialMaxPooling
	local ReLU= cudnn.ReLU
	local SpatialCrossMapLRN= cudnn.SpatialCrossMapLRN


	local alexnet_pretrained=load_alexnet()
	local features = nn.Sequential()
	for i=1, 22 do -- till Linear(4096 -> 4096)+ReLU+Dropout
		features:add(alexnet_pretrained.modules[i])
	end

	local object_branch = nn.Sequential()
		:add(nn.GradientReversal(params.grl_lambda))
		:add(nn.Linear(4096, 1000))
		:add(cudnn.SoftMax()) --BCE
		--:add(cudnn.LogSoftMax()) --NLL

	local activity_branch = nn.Sequential()
		:add(nn.Linear(4096, 1024))
		:add(ReLU())
		:add(nn.Dropout(0.5))
		:add(nn.Linear(1024, 504))
		:add(cudnn.LogSoftMax())

	-- increase the learning rate for new layers
	set_lr(object_branch.modules[2], 10) -- 10x lr_mult
	set_lr(activity_branch.modules[1], 10) -- 10x lr_mult
	set_lr(activity_branch.modules[4], 10) -- 10x lr_mult
	

	-- connect the two branches at the bottleneck
	local input=nn.Identity()()
	local common_feat=features(input)
	local obj_out=object_branch(common_feat)
	local act_out=activity_branch(common_feat)

	local model = nn.gModule({input}, {act_out, obj_out})
	model:cuda()

	return model

end
