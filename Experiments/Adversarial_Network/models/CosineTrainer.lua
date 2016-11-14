require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'optim'
require 'nngraph'
require 'rnn'
require 'nnlr'

---------------------------------------------------------------------
function cudafy(t_list)
	-- :cuda() a list of tensors (or a single tensor) 
	if torch.type(t_list)~='table'then
		return t_list:cuda()
	end

	for k,v in pairs(t_list) do
		t_list[k]=v:cuda()
	end
	return t_list
end


---------------------------------------------------------------------
local CosineTrainer = torch.class('CosineTrainer')

function CosineTrainer:__init(params)
	
	self.params=params
	self.epoch=1

	if params.load==nil or params.load=='nil' then
		require(self.params.model)
		self.model=build_model(self.params)
	else	
		self:load(params.load)
		self.params.cv_dir=params.cv_dir -- save models to new cv_dir
	end

	self.crit= nn.ParallelCriterion()
			:add(nn.ClassNLLCriterion(), 1)
			:add(nn.CosineEmbeddingCriterion(0.5), 1000)

	self.model:cuda()
	self.crit:cuda()

	self.sgdState = {
		learningRate  = params.base_lr,
		weightDecay   = params.base_wd,
		momentum    = 0.9,
		dampening   = 0,
		nesterov    = true
	}

	self.weights, self.gradients = self.model:getParameters()

end

function CosineTrainer:trainBatch(batchInputs, batchLabels, base_lr, base_wd)
	
	-- get layer-wise lr and update optim_state
	local base_lr= base_lr or self.params.base_lr
	local base_wd= base_wd or self.params.base_wd
	local learningRates, weightDecays = self.model:getOptimConfig(base_lr, base_wd)

	self.sgdState.learningRates = learningRates
    self.sgdState.weightDecays = weightDecays
    self.sgdState.learningRate = base_lr

	-- copy data to gpu
	batchInputs, batchLabels= cudafy(batchInputs), cudafy(batchLabels)
	batchLabels= batchLabels[1] -- ditch object labels

	--collectgarbage(); collectgarbage();
	self.model:training()
	self.gradients:zero()
	local y = self.model:forward(batchInputs)
	local cos_labels=torch.zeros(batchLabels:size(1)):fill(-1):cuda() --make everything go far away

	-- calculate loss normally
	local loss_val = self.crit:forward({y[1], {y[2], y[3]}}, {batchLabels, cos_labels})
	local nll_loss= self.crit.criterions[1].output
	local cos_loss= self.crit.criterions[2].output*self.crit.weights[2]

	local df_dw = self.crit:backward({y[1], {y[2], y[3]}}, {batchLabels, cos_labels})
	self.model:backward(batchInputs, {df_dw[1], unpack(df_dw[2])})

	optim.sgd(function()
		return loss_val, self.gradients
		end,
		self.weights,
		self.sgdState)
	return {loss_val, nll_loss, cos_loss}
end

function CosineTrainer:predict(input, target)
	self.model:evaluate()

	input= cudafy(input)
	local output = self.model:forward(input)

	local loss=-1
	if target~=nil then
		taget= cudafy(target)	
		target=target[1] -- ditch object labels
		local cos_labels=torch.zeros(target:size(1)):fill(-1):cuda() --make everything go far away
		loss= self.crit:forward({output[1], {output[2], output[3]}}, {target, cos_labels})
	end
	
	return {output[1]:float(), loss}
end

function CosineTrainer:save(epoch, loss)
	
	local checkpoint={}
	checkpoint.loss=loss
	checkpoint.epoch=epoch
	checkpoint.model=self.model:clone()
	checkpoint.model:clearState() -- save lots of memory
	checkpoint.params=self.params


	local outfile= self.params.cv_dir..string.format('/ep_%d_loss_%.6f.t7', epoch, loss)
	torch.save(outfile, checkpoint)
end


function CosineTrainer:load(filename)

	local checkpoint=torch.load(filename)
	self.model=checkpoint.model
	self.params=checkpoint.params
	self.epoch=checkpoint.epoch
	print (string.format('model loaded from %s', filename))
	
end

function CosineTrainer:set_grl_lambda(lambda)
	-- dummy function so train doesn't break
end


function set_lr(module, factor)
	if module.weight~=nil then
		module:learningRate('weight', factor)
			:learningRate('bias', factor)
			:weightDecay('weight', factor)
			:weightDecay('bias', factor)
	end
end

function CosineTrainer:reset_lr_mults()
	-- dummy function so train doesn't break
end

function CosineTrainer:set_lr_mults()
	-- dummy function so train doesn't break
end

function CosineTrainer:get_reps()
	-- dummy function so train doesn't break
end




