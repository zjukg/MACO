DATA=FB15K237
NUM_BATCH=1024
KERNEL=transe
MARGIN=2
LR=2e-5
NEG_NUM=16
VISUAL=allgen-vit
MS=0.4
POSTFIX=2.0-0.01-nocon

CUDA_VISIBLE_DEVICES=1 nohup python run_rsme.py -dataset=$DATA \
  -batch_size=$NUM_BATCH \
  -margin=$MARGIN \
  -epoch=1000 \
  -dim=128 \
  -save=./checkpoint/rsme/$DATA-rsme-$NUM_BATCH-$MARGIN-$LR-$VISUAL-$NEG_NUM-$MS \
  -img_grad=False \
  -img_dim=768 \
  -neg_num=$NEG_NUM \
  -kernel=$KERNEL \
  -visual=$VISUAL \
  -learning_rate=$LR \
  -postfix=$POSTFIX \
  -missing_rate=$MS > ./log/RSME$MS/$DATA-No-$MARGIN-$VISUAL-$NEG_NUM-$MS-$POSTFIX.txt &
