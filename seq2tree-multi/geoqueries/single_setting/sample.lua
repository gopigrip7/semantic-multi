require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'utils/SymbolsManager'
include "../utils/accuracy.lua"
include '../utils/MinibatchLoader.lua'

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

function do_generate(enc_w_list)
  -- encode
  for i = 1, #s do s[i]:zero() end
  -- reversed order
  -- local enc_w_list_withSE = shallowcopy(enc_w_list)
  local enc_w_list_withSE = {}
  for i = enc_w_list:size(1), 1, -1 do
    local t = enc_w_list[i]
    if t ~= 1 and t ~= 2 then
      table.insert(enc_w_list_withSE, t)
    end
  end
  table.insert(enc_w_list_withSE,1,word_manager:get_symbol_idx('<E>'))
  table.insert(enc_w_list_withSE,word_manager:get_symbol_idx('<S>'))

  local enc_s_top = transfer_data(torch.zeros(1, #enc_w_list_withSE, checkpoint.opt.rnn_size))
  for i = #enc_w_list_withSE, 1, -1 do
    local encoding_result = enc_rnn_unit:forward({transfer_data(torch.Tensor(1):fill(enc_w_list_withSE[i])), s})
    copy_table(s, encoding_result)

    enc_s_top[{{}, #enc_w_list_withSE-i+1, {}}]:copy(s[2*checkpoint.opt.num_layers])
  end

  -- decode
  local queue_decode = {}
  table.insert(queue_decode, {s=s, parent=0, child_index=1, t=seq2tree.Tree()})
  local head = 1
  while (head <= #queue_decode) and (head <= 100) do
    s = queue_decode[head].s
    local parent_h = s[2 * checkpoint.opt.num_layers]:clone()
    local t = queue_decode[head].t

    local prev_word
    if head == 1 then
      prev_word= float_transfer_data(torch.Tensor(1):fill(form_manager:get_symbol_idx('<S>')))
    else
      prev_word= float_transfer_data(torch.Tensor(1):fill(form_manager:get_symbol_idx('(')))
    end
    local i_child = 1
    while true do
      -- forward the rnn for next word
      local s_cur = dec_rnn_unit:forward({prev_word, s, parent_h})
      local prediction = dec_att_unit:forward({enc_s_top, s_cur[2*checkpoint.opt.num_layers]})
      copy_table(s, s_cur)
      
      -- log probabilities from the previous timestep
      local _, _prev_word = prediction:max(2)
      prev_word = _prev_word:resize(1)

      if (prev_word[1] == form_manager:get_symbol_idx('<E>')) or (t.num_children >= checkpoint.opt.dec_seq_length) then
        break
      elseif (prev_word[1] == form_manager:get_symbol_idx('<N>')) then
        table.insert(queue_decode, {s=clone_table(s), parent=head, child_index=i_child, t=seq2tree.Tree()})
        t:add_child(prev_word[1])
      else
        t:add_child(prev_word[1])
      end
      i_child = i_child + 1
    end
    head = head + 1
  end
  -- refine the root tree
  for i = #queue_decode, 2, -1 do
    local cur = queue_decode[i]
    queue_decode[cur.parent].t.children[cur.child_index] = cur.t
  end

  return queue_decode[1].t:to_list(form_manager)
end

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from the learned model')
cmd:text()
cmd:text('Options')
cmd:option('-model','model checkpoint to use for sampling')
cmd:option('-data_dir', '/disk/scratch_ssd/lidong/gen_review/books/', 'data directory')
cmd:option('-input', 'test', 'input data filename')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',0,' 0 to use max at each timestep (-beam_size = 1), 1 to sample at each timestep, 2 to beam search')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-beam_size',20,'beam size')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-display',1,'whether display on console')
cmd:option('-lang','','language to test (default = all)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
opt.output = opt.model .. '.' .. opt.lang .. '.sample'

-- initialize gpu/cpu
init_device(opt)

-- load the model checkpoint
checkpoint = torch.load(opt.model)
languages = checkpoint.opt.lang
lang_id = 0
for i=1,#languages do
  if opt.lang == languages[i] then
    lang_id = i
    print('evaluating ' .. opt.lang .. ' ' .. lang_id)
  end
end
enc_rnn_unit = checkpoint.enc_rnn_unit[lang_id]
dec_rnn_unit = checkpoint.dec_rnn_unit
local att_id = 1
if checkpoint.opt.att == 'single' then att_id = lang_id end
dec_att_unit = checkpoint.dec_att_unit[att_id]
-- put in eval mode so that dropout works properly
enc_rnn_unit:evaluate()
dec_rnn_unit:evaluate()
dec_att_unit:evaluate()

-- initialize the rnn state to all zeros
s = {}
local h_init = transfer_data(torch.zeros(1, checkpoint.opt.rnn_size))
for i = 1, checkpoint.opt.num_layers do
  -- c and h for all layers
  table.insert(s, h_init:clone())
  table.insert(s, h_init:clone())
end

-- initialize the vocabulary manager to display text
-- word_manager, form_manager = unpack(torch.load(path.join(opt.data_dir, 'map.t7')))
symbol_managers = torch.load(path.join(opt.data_dir, 'map.t7'))
local num_enc = #symbol_managers-1
form_manager = symbol_managers[num_enc+1]
word_manager = symbol_managers[lang_id]
-- load data
-- local data = torch.load(path.join(opt.data_dir, opt.input))
test_loader = seq2tree.MinibatchLoader()
test_loader:create2(opt, opt.input)
local data = test_loader:all_batch()

local f_out = torch.DiskFile(opt.output, 'w')
local reference_list = {}
local candidate_list = {}
for i = 1, #data do
  -- this will always be a batch of 1
  local enc_batch, enc_len_batch, dec_batch = unpack(data[i])
  local reference = dec_batch[1]
  local candidate = do_generate(enc_batch[lang_id][1])
  
  -- local x = data[i]
  -- local reference = x[2]
  -- local candidate = do_generate(x[1])

  table.insert(reference_list, reference)
  table.insert(candidate_list, candidate)

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
