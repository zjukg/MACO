DATA=FB15K237
NUM_BATCH=1024
KERNEL=transe
MARGIN=6
LR=2e-5
NEG_NUM=64
VISUAL=gen-vit
MS=0.8
POSTFIX=2.0-0.01

CUDA_VISIBLE_DEVICES=0 nohup python run_tbkgc.py -dataset=$DATA \
  -batch_size=$NUM_BATCH \
  -margin=$MARGIN \
  -epoch=1000 \
  -dim=128 \
  -save=./checkpoint/tbkgc/$DATA-New-$KERNEL-$NUM_BATCH-$MARGIN-$LR-$VISUAL-large-$MS \
  -img_grad=False \
  -img_dim=768 \
  -neg_num=$NEG_NUM \
  -kernel=$KERNEL \
  -visual=$VISUAL \
  -learning_rate=$LR \
  -postfix=$POSTFIX \
  -missing_rate=$MS > ./log/TBKGC$MS/$DATA-New-$KERNEL-5score-$MARGIN-$VISUAL-$MS-$POSTFIX.txt &
