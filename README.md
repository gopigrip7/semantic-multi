This repository contains the code for the ACL-17 paper: Neural Architectures for Multilingual Semantic Parsing, which implements multilingual extensions to Dong Li's seq2tree semantic parser ([github](https://github.com/donglixp/lang2logic)).

# Setup
To replicate the results in the paper, I recommend to use my Torch's package ([link](https://drive.google.com/file/d/0B6hvU8RdMvlWMnBkclN4dVFRUk0/view?usp=sharing)) and CUDA 7.5. Download and install using the following commands:
```bash
tar zxvf torch.tgz -C ~
cd ~/torch
bash install-deps
./clean.sh
./install.sh
luarocks install class
```

Clone the repo:
```bash
git clone https://gitlab.com/raymondhs/semantic-multi ~/semantic-multi
```

# Data
Download and extract the semantic datasets ([link](https://drive.google.com/file/d/0B6hvU8RdMvlWQ3ZaV3RYYXVpaTQ/view?usp=sharing)):
```bash
tar zxvf data.tgz -C ~/semantic-multi
```

# Running the code
The following scripts are provided: `run_monolingual.sh` (SINGLE), `run_ranking.sh` (RANKING), `run_multi_s.sh` (MULTI, single-source), and `run_multi_m.sh` (MULTI, multi-source). Example usage on GeoQuery:
```bash
# Set to between 1-3 (each with different random seed)
EXP_ID=1 

# Set which GPU to use
GPU_ID=0 

# Config files for each model
CONFIG_MONO=config/geo_mono.txt
CONFIG_MULTI_S=config/geo_multi_s.txt
CONFIG_MULTI_M=config/geo_multi_m.txt

# SINGLE
for lang in en de el th; do
  ./run_monolingual.sh seq2tree geoqueries attention $lang $CONFIG_MONO $EXP_ID $GPU_ID
done

# RANKING
./run_ranking.sh $GPU_ID geoqueries $CONFIG_MONO $EXP_ID

# MULTI (single-source)
ATT_S=shared # single or shared
./run_multi_s.sh seq2tree-multi geoqueries single_setting en,de,el,th $CONFIG_MULTI_S $ATT_S $EXP_ID

# MULTI (multi-source)
ATT_M=sent # word or sent
./run_multi_m.sh seq2tree-multi geoqueries multi_setting en,de,el,th $CONFIG_MULTI_M $ATT_M $EXP_ID
```
