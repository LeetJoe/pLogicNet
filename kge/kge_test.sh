#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=kge
SAVE_PATH=models

#The first four parameters must be provided
MODE=$1      # train/valid/test
MODEL=$2      # kge model name, RotatE/pRotatE/TransE/ComplEx/DistMult
DATA_PATH=$3      # 数据路径，或者数据集，FB15k/FB15k-237/wn18/wn18rr/countries_S{1,2,3}
GPU_DEVICE=$4      # 貌似是选择显卡，都是 0，或许 “0 1 2” 这样的值也是可以的

#Only used in training
BATCH_SIZE=$5      # 这个是 kge_batch todo 都是 0 ? 怎么回事
NEGATIVE_SAMPLE_SIZE=$6      # 对应 kge_neg, 取样大小，1024/512，FB15k* 用的都是 1024, wn18* 用的都是 512.
HIDDEN_DIM=$7      # 隐状态维度，256/1024(64/400), FB15k* 用的都是 256, wn18* 用的都是 1024.
GAMMA=$8      # 1000/500 FB15k* 用的都是 1000, wn18* 用的都是 500.
ALPHA=$9      # 24/12/9/6/0.1 应该是一个系数
LEARNING_RATE=${10}      # 1.0/0.5
MAX_STEPS=${11}      # kge_iters, 很小的数，1e-4/2e-4/5e-5，还有一些其它值
TEST_BATCH_SIZE=${12}      # 8e4/10e4/15e4 等值
WORKSPACE_PATH=${13}      # todo 这个在命令里好像没有对应？ best config 里面这个位置没有参数
TOP_K=${14}     # top k, 8/16

SAVE=$WORKSPACE_PATH/"$MODEL"

if [ $MODE == "train" ]
then

echo "Start Training......"

# todo 1. valid_steps 写死了？ 2. max_steps 是对应的那个很小的数吗？
# todo 最后面的那些 ${15~21} 是一些其它尚未整理的参数如 -de -dr -r 0.0005 --countries 需要进一步整理
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run_test.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path $DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -g $GAMMA -a $ALPHA -adv --record --valid_steps 50000 \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE --workspace_path $WORKSPACE_PATH --topk $TOP_K \
    ${15} ${16} ${17} ${18} ${19} ${20} ${21}

# valid 和 test 的参数比较简单
elif [ $MODE == "valid" ]
then

echo "Start Evaluation on Valid Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run_test.py --do_valid --cuda -init $SAVE
    
elif [ $MODE == "test" ]
then

echo "Start Evaluation on Test Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run_test.py --do_test --cuda -init $SAVE

else
   echo "Unknown MODE" $MODE
fi
