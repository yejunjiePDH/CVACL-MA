#export CUDA_VISIBLE_DEVICES=1

model_name=CVACL_MA

for pred_len in 24 36 48 60
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path national_illness.csv \
      --model_id custom_96_96_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 48 \
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
      --d_ff 128 \
     --batch_size 2 \
      --dropout 0 \
      --learning_rate 0.001 \
      --train_epochs 20 \
      --patience 3 \
      --n_heads 16 \
      --alpha 0.1 \
      --itr 1
done

