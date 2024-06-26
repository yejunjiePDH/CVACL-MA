#export CUDA_VISIBLE_DEVICES=1

model_name=CVACL_MA

#for pred_len in 96 192 336 720
#do
#python -u run.py \
#      --is_training 1 \
#      --root_path ./dataset/ETT-small/ \
#      --data_path ETTh2.csv \
#      --model_id ETTh2_96_96_$pred_len \
#      --model $model_name \
#      --data ETTh2 \
#      --features M \
#      --seq_len 96 \
#      --label_len 48 \
#      --pred_len $pred_len \
#      --e_layers 2 \
#      --d_layers 1 \
#      --factor 3 \
#      --enc_in 7 \
#      --dec_in 7 \
#      --c_out 7 \
#      --des 'Exp' \
#      --d_model 32 \
#      --d_ff 32 \
#     --batch_size 32 \
#      --dropout 0.3 \
#      --learning_rate 0.0005 \
#      --train_epochs 20 \
#      --patience 5 \
#      --n_heads 4 \
#      --alpha 0.4 \
#      --compare_baseline 5 \
#      --itr 1
#done


for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_96_96_$alpha$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 32 \
      --d_ff 32 \
     --batch_size 32 \
      --dropout 0.3 \
      --learning_rate 0.0005 \
      --train_epochs 20 \
      --patience 3 \
      --n_heads 8 \
      --alpha 0.1 \
      --compare_baseline 5 \
      --itr 1
done

