require 'torch'
require 'cunn'
require 'math'

local function main()
  network = nn.Sequential()
  
  inputs = 2
  hidden = 5
  outputs = 1
  
  network:add(nn.Linear(inputs, hidden))
  network:add(nn.Tanh())
  network:add(nn.Linear(hidden, outputs))
  network:cuda()
  
  print (network)
  
  lossFunc  = nn.SoftMarginCriterion()
  trainFunc = nn.StochasticGradient(network, lossFunc)
  trainFunc.learningRate = 0.1 

  bool2num = {[true]=1, [false]=0}

 
  for i = 1, 1000000 do
    local in1 = math.random(0, 1) 
    local in2 = math.random(0, 1)
    
    local out = torch.Tensor{bool2num[not(in1 == in2)]}
    out.cuda()
    
    local input = torch.Tensor{in1, in2}
    input.cuda()
    
    --print(in1, in2, out)
    result = network:forward(input)
    --print(result)
    loss = lossFunc:forward(result, out)
    print(loss)
    
    network:backward(input, lossFunc:backward(network.output, out))
    
    network:updateParameters(0.05, 0)
  end

end

main()
