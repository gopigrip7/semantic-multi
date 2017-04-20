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

function convert_to_string(manager, idx_list)
  local w_list = {}
  for i = 1, #idx_list do
    table.insert(w_list, manager:get_idx_symbol(idx_list[i]))
  end
  return table.concat(w_list, ' ')
end

function clone_tree(t)
  local new_t = seq2tree.Tree()
  new_t.parent = t.parent
  new_t.num_children = t.num_children
  new_t.children = {}
  for i = 1, #t.children do
    table.insert(new_t.children, t.children[i])
    if class.istype(t.children[i], 'seq2tree.Tree') then
      new_t.children[i].parent = new_t
    end
  end
  return new_t
end

function generate_next_word(hs, parent_h, prev_word_idx, enc_s_top)
  local prev_word = float_transfer_data(torch.Tensor(1):fill(prev_word_idx))
  -- forward the rnn for next word
  local s_cur = dec_rnn_unit:forward({prev_word, hs, parent_h})
  local prediction = dec_att_unit:forward({enc_s_top, s_cur[2*checkpoint.opt.num_layers]})

  local val, w_idx = prediction:sort(2, true)
  return w_idx:view(w_idx:nElement()):narrow(1,1,opt.beam_size):clone(),
    val:view(val:nElement()):narrow(1,1,opt.beam_size):clone(), clone_table(s_cur)
end

function do_generate(enc_w_list)
  -- encode
  for i = 1, #s do s[i]:zero() end
  -- reversed order
  local enc_w_list_withSE = shallowcopy(enc_w_list)
  table.insert(enc_w_list_withSE,1,word_manager:get_symbol_idx('<E>'))
  table.insert(enc_w_list_withSE,word_manager:get_symbol_idx('<S>'))
  local enc_s_top = transfer_data(torch.zeros(1, #enc_w_list_withSE, checkpoint.opt.rnn_size))
  for i = #enc_w_list_withSE, 1, -1 do
    local encoding_result = enc_rnn_unit:forward({transfer_data(torch.Tensor(1):fill(enc_w_list_withSE[i])), s})
    copy_table(s, encoding_result)

    enc_s_top[{{}, #enc_w_list_withSE-i+1, {}}]:copy(s[2*checkpoint.opt.num_layers])
  end

  -- decode
  local prb = 0
  if opt.sample == 0 or opt.sample == 1 then
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
        local _prev_prb, _prev_word = prediction:max(2)
        prev_word = _prev_word:resize(1)
        prb = prb + _prev_prb[1][1]

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
    return {{queue_decode[1].t:to_list(form_manager), prb}}
  else
    local beam_list = {{prb = 0, text_gen = {}, s = s}}
    local parent_h = s[2 * checkpoint.opt.num_layers]:clone()
    while true do
      local search_list = {}
      for i = 1, #beam_list do
        local h = beam_list[i]
        local last_word
        if #h.text_gen == 0 then
          last_word = form_manager:get_symbol_idx('<S>')
        else
          last_word = h.text_gen[#h.text_gen]
        end
        if last_word == form_manager:get_symbol_idx('<E>') then
          table.insert(search_list, h)
        else
          local w_new_list, p_list, s_cur = generate_next_word(h.s, parent_h, last_word, enc_s_top)
          for j = 1, w_new_list:nElement() do
            local w_new = w_new_list[j]
            local p = p_list[j]
            local text_gen_append = shallowcopy(h.text_gen)
            table.insert(text_gen_append, w_new)
            table.insert(search_list, {prb = h.prb + p, text_gen = text_gen_append, s = s_cur})
          end
        end
      end
      -- sort and get the new beam list
      table.sort(search_list, function(a,b) return a.prb > b.prb end)
      beam_list = table_topk(search_list, opt.beam_size)
      -- whether stop generating
      local is_all_end = true
      for i = 1, #beam_list do
        local h = beam_list[i]
        local last_word = h.text_gen[#h.text_gen]
        if last_word ~= form_manager:get_symbol_idx('<E>') and #h.text_gen < checkpoint.opt.dec_seq_length then
          is_all_end = false
          break
        end
      end
      if is_all_end then
        break
      end
    end

    local nbest = {}
    for i = 1, #beam_list do
      local s2 = clone_table(s)
      local h = beam_list[i]
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
          local prediction = dec_att_unit:forward({enc_s_top, s_cur[2*checkpoint.opt.num_layers]})
          copy_table(s2, s_cur)
          
          if head == 1 then
            prev_word = float_transfer_data(torch.Tensor(1):fill(h.text_gen[i_child]))
          else
            -- log probabilities from the previous timestep
            -- local _, _prev_word = prediction:max(2)
            -- prev_word = _prev_word:resize(1)
            local _prev_prb, _prev_word = prediction:max(2)
            prev_word = _prev_word:resize(1)
            h.prb = h.prb + _prev_prb[1][1]
          end

          if (prev_word[1] == form_manager:get_symbol_idx('<E>')) or (t.num_children >= checkpoint.opt.dec_seq_length) then
            break
          elseif (prev_word[1] == form_manager:get_symbol_idx('<N>')) then
            table.insert(queue_decode, {s=clone_table(s2), parent=head, child_index=i_child, t=seq2tree.Tree()})
            t:add_child(prev_word[1])
          else
            t:add_child(prev_word[1])
          end
          i_child = i_child + 1
          if head == 1 and i_child > #h.text_gen then
            break
          end
        end
        head = head + 1
      end
      -- refine the root tree
      for i = #queue_decode, 2, -1 do
        local cur = queue_decode[i]
        queue_decode[cur.parent].t.children[cur.child_index] = cur.t
      end
      table.insert(nbest, {queue_decode[1].t:to_list(form_manager), h.prb})
    end
    return nbest
  end
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
cmd:text()

-- parse input params
opt = cmd:parse(arg)
opt.output = opt.model .. '.nbest'

-- initialize gpu/cpu
init_device(opt)

-- load the model checkpoint
checkpoint = torch.load(opt.model)
enc_rnn_unit = checkpoint.enc_rnn_unit
dec_rnn_unit = checkpoint.dec_rnn_unit
dec_att_unit = checkpoint.dec_att_unit
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
word_manager, form_manager = unpack(torch.load(path.join(opt.data_dir, 'map.t7')))
-- load data
local data = torch.load(path.join(opt.data_dir, opt.input))

local f_out = torch.DiskFile(opt.output, 'w')
local reference_list = {}
local candidate_list = {}
for i = 1, #data do
  local x = data[i]
  local reference = x[2]
  -- local candidate = do_generate(x[1])
  local nbest = do_generate(x[1])
  print("Decoding sentence " .. i .. " " .. convert_to_string(word_manager, x[1]))
  for j = 1, #nbest do
    local candidate, cand_prb = unpack(nbest[j])
    local cand_str = convert_to_string(form_manager, candidate)
    f_out:writeString(cand_str..'\t'..cand_prb..'\n')
  end
  f_out:writeString('\n')
   
  if i % 100 == 0 then
    collectgarbage()
  end
end

f_out:close()
