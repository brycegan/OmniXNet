python ./run.py \
 --task_name super-resolution \
 --is_training 6 \
 --model omnixnet \
 --data custom \
 --gpu 4 \
 --learning_rate 0.00003 \
 --des lr3e-5_MAE_base \
 --batch_size 32 \
 --img_size 128 \
 --num_classes 7 \
 --in_chans 11 \
 --num_workers 4 \
 --loss_regression MAE \
#  --use_checkpoint True


