set -e -x
# ./run_multi_s.sh seq2tree-multi [geoqueries|atis] single_setting lang1,lang2,...,langN param_file [single|shared] param_id

if [ -z $8 ] ; then
  GPU_ID=0
else
  GPU_ID=$8
fi

exp_name="$2"-multi-s-"$4"-"$6"Att

if [ $3 = "single_setting" ] || [ $3 = "multi_setting" ]; then
  PWD_DIR=$(pwd)
  WORK_DIR=$(dirname "$(readlink -f "$0")")/$1/$2
  INIT_DIR=$(dirname "$(readlink -f "$0")")/seq2tree/$2/dump_attention
  mkdir -p $WORK_DIR/data/$exp_name
  cd $WORK_DIR
  th data.lua -data_dir $WORK_DIR/data/multi -out_dir $WORK_DIR/data/$exp_name -lang $4
  prefix=0
  while read -r param; do
    prefix=$((prefix+1))
    if [ $prefix -eq $7 ] ; then
      DUMP_DIR=$WORK_DIR/dump_$3/"$exp_name"-$prefix
      mkdir -p $DUMP_DIR
      logfile=$PWD_DIR/logs/$exp_name/train-$prefix.log
      mkdir -p $PWD_DIR/logs/$exp_name
      CUDA_VISIBLE_DEVICES=$GPU_ID th $3/main.lua -data_dir $WORK_DIR/data/$exp_name -checkpoint_dir $DUMP_DIR \
      -lang $4 -att $6 \
      -init_weight_dir $INIT_DIR \
      $param > $logfile 2>&1
      for lang in `echo $4 | awk -F, '{for (i=1;i<=NF;i++)print $i}'`; do
        CUDA_VISIBLE_DEVICES=$GPU_ID th $3/sample.lua -data_dir $WORK_DIR/data/$exp_name -model $DUMP_DIR/model.t7 -lang $lang -sample 0 -display 0 >> $logfile 2>&1
      done
    fi
  done < $PWD_DIR/$5
  cd $PWD_DIR
fi
