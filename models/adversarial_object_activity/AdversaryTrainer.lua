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
local AdversaryTrainer = torch.class('AdversaryTrainer')

function AdversaryTrainer:__init(params)
	
	self.params=params
	self.epoch=1

	if params.load==nil or params.load=='nil' then
		require(self.params.model)
		self.model=build_model(self.params)
	else	
		self:load(params.load)
		self.params.cv_dir=params.cv_dir -- keep new cv_dir info
	end

	self.crit=nn.ParallelCriterion()
				:add(nn.ClassNLLCriterion(),1) -- activity branch
				:add(nn.BCECriterion(),100) -- object branch

	-- val crit is only on activity labels
	self.val_crit=nn.ParallelCriterion()
				:add(nn.ClassNLLCriterion(),1)
				:add(nn.BCECriterion(),100)

	-- mask for NLL
	--self.crit=nn.MaskZeroCriterion(self.crit, 1)
	--self.val_crit=nn.MaskZeroCriterion(self.val_crit, 1)
	

	self.model:cuda()
	self.crit:cuda()
	self.val_crit:cuda()

	self.sgdState = {
		learningRate  = params.base_lr,
		weightDecay   = params.base_wd,
		momentum    = 0.9,
		dampening   = 0,
		nesterov    = true
	}

	self.weights, self.gradients = self.model:getParameters()

end

function make_mask(labels, output)
	
	-- for BCE
	local mask=torch.repeatTensor(1-torch.eq(labels:sum(2),0),1,output:size(2)):cuda()
	
	-- for NLL
	--local mask=torch.repeatTensor(torch.gt(labels,0),output:size(2), 1):transpose(1,2):cuda()

	return mask
end
	

function AdversaryTrainer:trainBatch(batchInputs, batchLabels, base_lr, base_wd)
	
	-- get layer-wise lr and update optim_state
	local base_lr= base_lr or self.params.base_lr
	local base_wd= base_wd or self.params.base_wd
	local learningRates, weightDecays = self.model:getOptimConfig(base_lr, base_wd)

	self.sgdState.learningRates = learningRates
    self.sgdState.weightDecays = weightDecays
    self.sgdState.learningRate = base_lr

	-- copy data to gpu
	batchInputs, batchLabels= cudafy(batchInputs), cudafy(batchLabels)

	--collectgarbage(); collectgarbage();
	self.model:training()
	self.gradients:zero()
	local y = self.model:forward(batchInputs)

	
	-- mask object predictions to prevent learning
	local mask=make_mask(batchLabels[2], y[2])
	y[2]=torch.cmul(y[2],mask)

	-- calculate loss normally
	local loss_val = self.crit:forward(y, batchLabels)
	local df_dw = self.crit:backward(y, batchLabels)
	self.model:backward(batchInputs, df_dw)

	optim.sgd(function()
		return loss_val, self.gradients
		end,
		self.weights,
		self.sgdState)
	return loss_val
end

function AdversaryTrainer:predict(input, target)
	self.model:evaluate()

	local output = self.model:forward(cudafy(input))

	local loss=-1
	if target~=nil then
		local mask=make_mask(target[2], output[2])
		output[2]=torch.cmul(output[2],mask)
		loss= self.val_crit:forward(output, cudafy(target))
	end
	
	if torch.type(output)=='table' then
		return {output[1]:float(), loss}
	end

	return {output:float(), loss}
	
end

function AdversaryTrainer:save(epoch, loss)
	
	local checkpoint={}
	checkpoint.loss=loss
	checkpoint.epoch=epoch
	checkpoint.model=self.model:clone()
	checkpoint.model:clearState() -- save lots of memory
	checkpoint.params=self.params


	local outfile= self.params.cv_dir..string.format('/ep_%d_loss_%.6f.t7', epoch, loss)
	torch.save(outfile, checkpoint)
end


function AdversaryTrainer:load(filename)

	local checkpoint=torch.load(filename)
	self.model=checkpoint.model
	self.params=checkpoint.params
	self.epoch=checkpoint.epoch
	print (string.format('model loaded from %s', filename))
	
end

function AdversaryTrainer:set_grl_lambda(lambda)
	self.model.modules[4].modules[1].lambda= lambda
end


function set_lr(module, factor)
	if module.weight~=nil then
		module:learningRate('weight', factor)
			:learningRate('bias', factor)
			:weightDecay('weight', factor)
			:weightDecay('bias', factor)
	end
end

function AdversaryTrainer:reset_lr_mults()
	set_lr(self.model.modules[4].modules[2], 1) -- obj lin
	set_lr(self.model.modules[3].modules[1], 1) -- act lin 1
	set_lr(self.model.modules[3].modules[4], 1) -- act lin 2
end


function AdversaryTrainer:get_reps()
	-- get activity activations after ReLU from feature branch
	local activations= self.model.modules[2].modules[21].output
	activations=activations:float()
	return activations
end




