#export CUDA_VISIBLE_DEVICES=1

model_name=CVACL_MA

#for compare_baseline in $(seq 14 21)
for pred_len in 96 192
do
python -u run.py \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_96_$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 1 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --dropout 0.1 \
        --train_epochs 20 \
        --patience 3 \
        --n_heads 8 \
        --alpha 0.1 \
        --batch_size 16 \
        --compare_baseline -1 \
        --learning_rate 0.0005
done

for pred_len in 336
do
python -u run.py \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_96_$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 1 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 256 \
        --itr 1 \
        --dropout 0.2 \
        --train_epochs 20 \
        --patience 3 \
        --n_heads 8 \
        --alpha 0.1 \
        --batch_size 32 \
        --compare_baseline -1 \
        --learning_rate 0.001
done



for pred_len in 720
do
python -u run.py \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_96_$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 1 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --d_model 256 \
        --d_ff 512 \
        --itr 1 \
        --dropout 0.1 \
        --train_epochs 20 \
        --patience 3 \
        --n_heads 8 \
        --alpha 0.1 \
        --batch_size 16 \
        --compare_baseline -1 \
        --learning_rate 0.001
done
