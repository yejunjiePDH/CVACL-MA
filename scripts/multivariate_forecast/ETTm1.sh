#export CUDA_VISIBLE_DEVICES=1

model_name=CVACL_MA

#for pred_len in 96 192
for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm1.csv \
      --model_id ETTm1_96_96_$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 1 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 128 \
      --d_ff 256 \
     --batch_size 32 \
      --dropout 0.1 \
      --learning_rate 0.0005 \
      --train_epochs 20 \
      --patience 5 \
      --n_heads 8 \
      --compare_baseline 4 \
      --alpha 0.3 \
      --itr 1
done

# learning_rate 0.0003 0.0004
