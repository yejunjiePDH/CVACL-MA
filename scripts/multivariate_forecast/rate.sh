#export CUDA_VISIBLE_DEVICES=1

model_name=CVACL_MA

for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path exchange_rate.csv \
      --model_id exchange_rate_96_96_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --d_model 128 \
      --d_ff 128 \
      --batch_size 32 \
      --dropout 0.1 \
      --learning_rate 0.0001 \
      --train_epochs 20 \
      --patience 3 \
      --n_heads 8 \
      --alpha 0.2 \
      --compare_baseline -1 \
      --itr 1
done

