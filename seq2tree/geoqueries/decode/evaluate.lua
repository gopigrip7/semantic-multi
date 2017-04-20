require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'utils/SymbolsManager'
include "../utils/accuracy.lua"
local class = require 'class'

function transfer_data(x)
  if opt.gpuid >= 0 then
    return x:cuda()
  end
  return x
end

function float_transfer_data(x)
  if opt.gpuid>=0 then
    return x:float():cuda()
  end
  return x
end

function convert_to_string(idx_list)
  local w_list = {}
  for i = 1, #idx_list do
    table.insert(w_list, form_manager:get_idx_symbol(idx_list[i]))
  end
  return table.concat(w_list, ' ')
end

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from the learned model')
cmd:text()
cmd:text('Options')
cmd:option('-model','model checkpoint to use for sampling')
cmd:option('-data_dir', '/disk/scratch_ssd/lidong/gen_review/books/', 'data directory')
cmd:option('-input', 'test.t7', 'input data filename')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',0,' 0 to use max at each timestep (-beam_size = 1), 1 to sample at each timestep, 2 to beam search')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-beam_size',10,'beam size')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-display',1,'whether display on console')
cmd:option('-prediction','','prediction file')
cmd:option('-output_prefix','','')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
opt.output = opt.output_prefix .. '.eval'

word_manager, form_manager = unpack(torch.load(path.join(opt.data_dir, 'map.t7')))
-- load data
local data = torch.load(path.join(opt.data_dir, opt.input))

local candidate_list = {}
print('loading prediction file...')
local f = torch.DiskFile(opt.prediction, 'r', true)
f:clearError()
local rawdata = f:readString('*l')
while (not f:hasError()) do
  local tokens = rawdata:strip():split(' ')
  table.insert(candidate_list, form_manager:get_symbol_idx_for_list(tokens))
  -- read next line
  rawdata = f:readString('*l')
end
f:close()

local f_out = torch.DiskFile(opt.output, 'w')
local reference_list = {}
for i = 1, #data do
  local x = data[i]
  local reference = x[2]
  local candidate = candidate_list[i]

  table.insert(reference_list, reference)

  local ref_str = convert_to_string(reference)
  local cand_str = convert_to_string(candidate)
  -- print to console
  if opt.display > 0 then
    print(ref_str)
    print(cand_str)
    print(' ')
  end
  -- write to file
  f_out:writeString(ref_str)
  f_out:writeString('\n')
  f_out:writeString(cand_str)
  f_out:writeString('\n\n')
   
  if i % 100 == 0 then
    collectgarbage()
  end
end

-- compute evaluation metric
local val_acc = compute_tree_accuracy(candidate_list, reference_list, form_manager)
print('ACCURACY = ' .. val_acc)
f_out:writeString('ACCURACY = ' .. val_acc)
f_out:writeString('\n')

f_out:close()
