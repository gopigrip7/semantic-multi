require "utils/SymbolsManager.lua"
require "utils/utils.lua"

function process_train_data(opt)
  require('pl.stringx').import()
  require 'pl.seq'

  local timer = torch.Timer()
  
  local data = {}
  local symbol_managers = {}

  for i = 1, #opt.lang do
    local lang = opt.lang[i]
    local word_manager = SymbolsManager(true)
    word_manager:init_from_file(path.join(opt.data_dir, lang, 'vocab.q.txt'), opt.min_freq, opt.max_vocab_size)
    -- local form_manager = SymbolsManager(true)
    -- form_manager:init_from_file(path.join(opt.data_dir, lang, 'vocab.f.txt'), 0, opt.max_vocab_size)

    table.insert(symbol_managers, word_manager)

    print('loading text file...'..lang)
    local f = torch.DiskFile(path.join(opt.data_dir, lang, opt.train .. '.txt'), 'r', true)
    f:clearError()
    local rawdata = f:readString('*l')
    local j = 1
    while (not f:hasError()) do
      local l_list = rawdata:strip():split('\t')
      local w_list = word_manager:get_symbol_idx_for_list(l_list[1]:split(' '))
      -- local r_list = form_manager:get_symbol_idx_for_list(l_list[2]:split(' '))
      if i == 1 then data[j] = {} end
      table.insert(data[j], w_list)
      -- read next line
      rawdata = f:readString('*l')
      j = j + 1
    end
    f:close()

    collectgarbage()
  end

  -- obtain logic form using first language
  local lang = opt.lang[1]
  local form_manager = SymbolsManager(true)
  form_manager:init_from_file(path.join(opt.data_dir, lang, 'vocab.f.txt'), 0, opt.max_vocab_size)
  table.insert(symbol_managers, form_manager)
  local f = torch.DiskFile(path.join(opt.data_dir, lang, opt.train .. '.txt'), 'r', true)
  f:clearError()
  local rawdata = f:readString('*l')
  local j = 1
  while (not f:hasError()) do
    local l_list = rawdata:strip():split('\t')
    local r_list = form_manager:get_symbol_idx_for_list(l_list[2]:split(' '))
    table.insert(data[j], r_list)
    table.insert(data[j], convert_to_tree(r_list, 1, #r_list, form_manager))
    -- read next line
    rawdata = f:readString('*l')
    j = j + 1
  end
  f:close()

  collectgarbage()

  -- save output preprocessed files
  local out_mapfile = path.join(opt.out_dir, 'map.t7')
  print('saving ' .. out_mapfile)
  torch.save(out_mapfile, symbol_managers)

  collectgarbage()

  local out_datafile = path.join(opt.out_dir, opt.train .. '.t7')
  print('saving ' .. out_datafile)
  torch.save(out_datafile, data)

  collectgarbage()
end

function serialize_data(opt, name)
  require('pl.stringx').import()
  require 'pl.seq'
  
  local timer = torch.Timer()
  
  local symbol_managers = torch.load(path.join(opt.out_dir, 'map.t7'))
  local form_manager = symbol_managers[#symbol_managers]

  local data = {}

  for i = 1, #opt.lang do
    local lang = opt.lang[i]
    local fn = path.join(opt.data_dir, lang, name .. '.txt')
    if not path.exists(fn) then
      print('no file: ' .. fn)
      return nil
    end

    local word_manager = symbol_managers[i]
    local lang = opt.lang[i]
    print('loading text file...'..lang)
    local f = torch.DiskFile(fn, 'r', true)
    f:clearError()
    local rawdata = f:readString('*l')
    local j = 1
    while (not f:hasError()) do
      local l_list = rawdata:strip():split('\t')
      local w_list = word_manager:get_symbol_idx_for_list(l_list[1]:split(' '))
      local r_list = form_manager:get_symbol_idx_for_list(l_list[2]:split(' '))
      if i == 1 then data[j] = {} end
      table.insert(data[j], w_list)
      if i == #opt.lang then
        table.insert(data[j], r_list)
        table.insert(data[j], convert_to_tree(r_list, 1, #r_list, form_manager))
      end
      -- read next line
      rawdata = f:readString('*l')
      j = j + 1
    end
    f:close()

    collectgarbage()
  end

  -- save output preprocessed files
  local out_datafile = path.join(opt.out_dir, name .. '.t7')

  print('saving ' .. out_datafile)
  torch.save(out_datafile, data)
end

local cmd = torch.CmdLine()
cmd:option('-data_dir', '', 'data directory')
cmd:option('-train', 'train', 'train data path')
cmd:option('-dev', 'dev', 'dev data path')
cmd:option('-test', 'test', 'test data path')
cmd:option('-min_freq', 2, 'minimum word frequency')
cmd:option('-max_vocab_size', 15000, 'maximum vocabulary size')
cmd:option('-lang', 'en', 'list of languages separated by comma')

cmd:option('-out_dir', '', 'save directory')

cmd:text()
opt = cmd:parse(arg)

local languages = opt.lang:strip():split(',')
opt.lang = {}
for i = 1, #languages do
  table.insert(opt.lang, languages[i])
end
process_train_data(opt)
serialize_data(opt, opt.dev)
serialize_data(opt, opt.test)
