#export CUDA_VISIBLE_DEVICES=0

model_name=CVACL_MA

for pred_len in 96
do
python -u run.py \
        --is_training 1 \
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id ECL_96_96_$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --des 'Exp' \
        --d_model 256 \
        --d_ff 512 \
        --batch_size 4 \
        --dropout 0 \
        --learning_rate 0.001 \
        --patience 3 \
        --n_heads 16 \
        --alpha 0.5 \
        --train_epochs 20 \
        --compare_baseline -1 \
        --itr 1
done

# 0.6 以上相关有40个，0.7 以上相关有2个， # 0.5 上有138 # 0.4 以上有241  # 0.3 以上283 个 # 0.2 以上301个  # 0.1 以上310个  # 0以上314个

for pred_len in 192 336
do
python -u run.py \
        --is_training 1 \
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id ECL_96_96_$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --des 'Exp' \
        --d_model 256 \
        --d_ff 256 \
        --batch_size 4 \
        --dropout 0.1 \
        --learning_rate 0.001 \
        --patience 3 \
        --n_heads 16 \
        --alpha 0.5 \
        --train_epochs 20 \
        --compare_baseline -1 \
        --itr 1
done


for pred_len in 720
do
python -u run.py \
        --is_training 1 \
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id ECL_96_96_$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --des 'Exp' \
        --d_model 256 \
        --d_ff 256 \
        --batch_size 4 \
        --dropout 0 \
        --learning_rate 0.001 \
        --patience 3 \
        --n_heads 16 \
        --alpha 0.5 \
        --train_epochs 20 \
        --compare_baseline -1 \
        --itr 1
done
