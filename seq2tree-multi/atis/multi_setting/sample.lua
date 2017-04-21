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

function do_generate(enc_batch, enc_len_batch, num_enc)
  -- ship batch data to gpu
  local enc_max_len = 0
  for n = 1, num_enc do
    enc_batch[n] = float_transfer_data(enc_batch[n])
    if enc_batch[n]:size(2) > enc_max_len then
      enc_max_len = enc_batch[n]:size(2)
    end
  end

  -- encode
  for n = 1, num_enc do
    for i = 1, #s[n] do s[n][i]:zero() end
  end

  for j = 1, 2 * checkpoint.opt.num_layers do
    last_s[j]:zero()
  end

  local enc_s_top_list = {}
  for n = 1, num_enc do
    local enc_s_top = transfer_data(torch.zeros(1, enc_max_len, checkpoint.opt.rnn_size))
    table.insert(enc_s_top_list, enc_s_top)
  end

  for i = 1, enc_max_len do
    local inputs = {}
    for n = 1, num_enc do
      table.insert(inputs, enc_batch[n][{{}, i}])
      table.insert(inputs, s[n])
    end
    local tmp = enc_rnn_unit:forward(inputs)
    if num_enc == 1 then
      copy_table(s[1], tmp)
    else
      for n = 1, num_enc do
        copy_table(s[n], tmp[n])
      end
    end
    
    for n = 1, num_enc do
      local seq_len = enc_len_batch[n][1]

      -- for attention computation
      local enc_s_top = enc_s_top_list[n]
      enc_s_top[{{}, i, {}}]:copy(s[n][2*checkpoint.opt.num_layers])

      -- for decoder's initialization
      if i == seq_len then
        for j = 1, 2 * checkpoint.opt.num_layers do
          last_s[j][1][n]:copy(s[n][j])
        end
      end
    end
  end

  for j = 1, 2*checkpoint.opt.num_layers do
    s2[j]:copy(dec_init_unit:forward(last_s[j]))
  end

  -- decode
  local queue_decode = {}
  table.insert(queue_decode, {s=s2, parent=0, child_index=1, t=seq2tree.Tree()})
  local head = 1
  while (head <= #queue_decode) and (head <= 100) do
    s2 = queue_decode[head].s
    local parent_h = s2[2 * checkpoint.opt.num_layers]:clone()
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
      local s_cur = dec_rnn_unit:forward({prev_word, s2, parent_h})
      enc_s_top_list[num_enc+1] = s_cur[2*checkpoint.opt.num_layers] -- last decoder's state
      local prediction = dec_att_unit:forward(enc_s_top_list)
      copy_table(s2, s_cur)
      
      -- log probabilities from the previous timestep
      local _, _prev_word = prediction:max(2)
      prev_word = _prev_word:resize(1)

      if (prev_word[1] == form_manager:get_symbol_idx('<E>')) or (t.num_children >= checkpoint.opt.dec_seq_length) then
        break
      elseif (prev_word[1] == form_manager:get_symbol_idx('<N>')) then
        table.insert(queue_decode, {s=clone_table(s2), parent=head, child_index=i_child, t=seq2tree.Tree()})
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
cmd:option('-data_dir', '', 'data directory')
cmd:option('-input', 'test', 'input data filename')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',0,' 0 to use max at each timestep (-beam_size = 1), 1 to sample at each timestep, 2 to beam search')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-beam_size',20,'beam size')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-display',1,'whether display on console')
cmd:option('-output','','output file')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
if opt.output == '' then
  opt.output = opt.model .. '.sample'
end

-- initialize gpu/cpu
init_device(opt)

-- load the model checkpoint
checkpoint = torch.load(opt.model)
enc_rnn_unit = checkpoint.enc_rnn_unit
dec_rnn_unit = checkpoint.dec_rnn_unit
dec_att_unit = checkpoint.dec_att_unit
dec_init_unit = transfer_data(nn.Max(2,1))

-- put in eval mode so that dropout works properly
enc_rnn_unit:evaluate()
dec_rnn_unit:evaluate()
dec_att_unit:evaluate()

-- initialize the vocabulary manager to display text
symbol_managers = torch.load(path.join(opt.data_dir, 'map.t7'))
local num_enc = #symbol_managers-1
form_manager = symbol_managers[num_enc+1]

-- initialize the rnn state to all zeros
s = {}
local h_init = transfer_data(torch.zeros(1, checkpoint.opt.rnn_size))
for n = 1, num_enc do
  s[n] = {}
  for i = 1, checkpoint.opt.num_layers do
    -- c and h for all layers
    table.insert(s[n], h_init:clone())
    table.insert(s[n], h_init:clone())
  end
end

last_s = {} -- depends on a language-specific sequence length
for i = 1, 2 * checkpoint.opt.num_layers do
  last_s[i] = transfer_data(torch.zeros(1, num_enc, checkpoint.opt.rnn_size))
end

s2 = {}
for i = 1, checkpoint.opt.num_layers do
  -- c and h for all layers
  table.insert(s2, h_init:clone())
  table.insert(s2, h_init:clone())
end

-- load data
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
  local candidate = do_generate(enc_batch, enc_len_batch, num_enc)

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
