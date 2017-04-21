set -e -x
gpuid=$1
data=$2
params_file=$3
param_id=$4
shift 4
if [ $data = "geoqueries" ]; then
    languages="en de el th"
elif [ $data = "atis" ]; then
    languages="en id zh"
fi

PWD_DIR=$(pwd)
for lang in $languages; do
    mkdir -p logs/$data-$lang
    LOG_DIR=$PWD_DIR/logs/$data-$lang
    WORK_DIR=$(dirname "$(readlink -f "$0")")/seq2tree/$data
    cd $WORK_DIR
    th data.lua -data_dir $WORK_DIR/data/$lang > $LOG_DIR/train-$lang.log
    cd $PWD_DIR
    prefix=0
    while read -r param; do
        prefix=$((prefix+1))
        if [ $prefix -eq $param_id ] ; then
            ./decode.sh seq2tree $data attention $lang $params_file $prefix $gpuid > logs/$data-$lang/decode-$lang-$prefix.log 2>&1
        fi
    done < $params_file
done

WORK_DIR=$(dirname "$(readlink -f "$0")")/seq2tree/$data
DECODE_DIR=$WORK_DIR/decode
DUMP_DECODE_DIR=$WORK_DIR/dump_decode
DUMP_DIR=$WORK_DIR/dump_attention
mkdir -p $DUMP_DECODE_DIR
if [ $data = "geoqueries" ]; then
    langset="en,de,el en,de,th en,el,th de,el,th en,de,el,th"
elif [ $data = "atis" ]; then
    langset="en,id en,zh id,zh en,id,zh"
fi
prefix=0
while read -r param; do
    prefix=$((prefix+1))
    if [ $prefix -eq $param_id ] ; then
        for langs in $langset; do
            files=""
            for lang in `echo $langs | awk -F, '{for (i=1;i<=NF;i++)print $i}'`; do
                files="$files $DUMP_DIR/$lang-$prefix/model.t7.nbest"
            done
            output=$DUMP_DECODE_DIR/$langs-$prefix.out
            python $DECODE_DIR/aggregate.py $files $normalize > $output
            
            mkdir -p logs/$data-ranking-$langs
            logfile=$PWD_DIR/logs/$data-ranking-$langs/aggregate-$prefix.log

            cd $WORK_DIR
            CUDA_VISIBLE_DEVICES=$gpuid th decode/evaluate.lua -data_dir $WORK_DIR/data/en -input test.orig.t7 -output_prefix $DUMP_DECODE_DIR/$langs-$prefix -prediction $output -display 0 > $logfile 2>&1
            cd $PWD_DIR
        done
    fi
done < $params_file
