require 'cudnn'
require 'nngraph'
require 'nnlr'
require 'paths'

function load_alexnet()

	local pretrain_dir='../../models/pretrained/'

	local caffe_t7=pretrain_dir..'bvlc_reference_caffenet.t7'
	if paths.filep(caffe_t7) then
		local model= torch.load(caffe_t7)
		return model
	end

	require 'loadcaffe'
	
	local deploy= deploy or pretrain_dir..'deploy.prototxt'
	local weights=weights or pretrain_dir..'bvlc_reference_caffenet.caffemodel'

	local model = loadcaffe.load(deploy, weights, 'cudnn')
	torch.save(caffe_t7, model)
	print 'bvlc model saved as .t7'

	return model
end

function load_activity_branch()
	
	require '../../models/adversary'
	local base_cv=torch.load('cv/adversary_activity/ep_10_loss_3.698637.t7')
	local trunk=base_cv.model.modules[2]
	trunk:remove() -- throw off the dropout layer

	return trunk

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

	local object_branch=load_alexnet()
	object_branch:remove()
	object_branch:remove()
	object_branch:remove()

	--local activity_branch=object_branch:clone()

	local activity_branch=load_activity_branch()
	local activity_classifier=nn.Sequential()
				:add(nn.Dropout(0.5))
				:add(nn.Linear(4096, params.num_acts))
				:add(cudnn.LogSoftMax())


	-- freeze all weights of object branch
	for i=1, #object_branch.modules do
		set_lr(object_branch.modules[i], 0)
	end


	local input=nn.Identity()()
	local obj_rep=object_branch(input)
	local act_rep=activity_branch(input)
	local act_pred=activity_classifier(act_rep)

	local model=nn.gModule({input},{act_pred, act_rep, obj_rep}) 

	model:cuda()
	return model

end
