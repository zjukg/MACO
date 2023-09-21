DATA=FB15K237
NUM_BATCH=1024
KERNEL=transe
MARGIN=6
LR=2e-5
NEG_NUM=32
VISUAL=random-vit
MS=0.6
POSTFIX=2.0-0.01

CUDA_VISIBLE_DEVICES=0 nohup python run_ikrl.py -dataset=$DATA \
  -batch_size=$NUM_BATCH \
  -margin=$MARGIN \
  -epoch=1000 \
  -dim=128 \
  -save=./checkpoint/ikrl/$DATA-New-$KERNEL-$NUM_BATCH-$MARGIN-$LR-$VISUAL-large-$MS \
  -img_grad=False \
  -img_dim=768 \
  -neg_num=$NEG_NUM \
  -kernel=$KERNEL \
  -visual=$VISUAL \
  -learning_rate=$LR \
  -postfix=$POSTFIX \
  -missing_rate=$MS > ./log/IKRL$MS/$DATA-$KERNEL-4score-$MARGIN-$VISUAL-$MS-$POSTFIX.txt &
