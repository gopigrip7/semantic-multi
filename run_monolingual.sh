set -e -x
# ./run_monolingual.sh seq2tree [geoqueries|atis] attention lang param_file param_id

if [ -z $7 ] ; then
  GPU_ID=0
else
  GPU_ID=$7
fi

mkdir -p logs/$2-$4

if [ $3 = "lstm" ] || [ $3 = "attention" ] ; then
  PWD_DIR=$(pwd)
  WORK_DIR=$(dirname "$(readlink -f "$0")")/$1/$2
  cd $WORK_DIR
  th data.lua -data_dir $WORK_DIR/data/$4
  prefix=0
  while read -r param; do
    prefix=$((prefix+1))
    if [ $prefix -eq $6 ] ; then
      DUMP_DIR=$WORK_DIR/dump_$3/$4-$prefix
      mkdir -p $DUMP_DIR
      logfile=$PWD_DIR/logs/$2-$4/train-$prefix.log
      tune_logfile=$PWD_DIR/logs/$2-$4/tune-$prefix.log
      CUDA_VISIBLE_DEVICES=$GPU_ID th $3/main.lua -data_dir $WORK_DIR/data/$4 -checkpoint_dir $DUMP_DIR \
      $param > $logfile 2>&1
      CUDA_VISIBLE_DEVICES=$GPU_ID th $3/sample.lua -data_dir $WORK_DIR/data/$4 -model $DUMP_DIR/model.t7 -sample 0 -display 0 >> $logfile 2>&1
    fi
  done < $PWD_DIR/$5
  cd $PWD_DIR
fi
