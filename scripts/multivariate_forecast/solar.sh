#export CUDA_VISIBLE_DEVICES=1

# 0.7以上137个   # 0.8以上136个   # 0.9以上53 个

model_name=CVACL_MA

for pred_len in 96
do
python -u run.py \
        --is_training 1 \
        --root_path ./dataset/solar/ \
        --data_path solar_AL.csv \
        --model_id solar_96_96_$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --d_layers 1 \
        --factor 3 \
        --enc_in 137 \
        --dec_in 137 \
        --c_out 137 \
        --des 'Exp' \
        --e_layers 2 \
        --d_model 256 \
        --d_ff 256 \
        --dropout 0.1 \
        --n_heads 4 \
        --train_epochs 20 \
        --patience 3 \
        --alpha 0 \
        --batch_size 16 \
        --train_epochs 20 \
        --itr 1 \
        --compare_baseline -1 \
        --learning_rate 0.001
done



for pred_len in 192
do
python -u run.py \
        --is_training 1 \
        --root_path ./dataset/solar/ \
        --data_path solar_AL.csv \
        --model_id solar_96_96_$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --d_layers 1 \
        --factor 3 \
        --enc_in 137 \
        --dec_in 137 \
        --c_out 137 \
        --des 'Exp' \
        --e_layers 2 \
        --d_model 128 \
        --d_ff 512 \
        --dropout 0.1 \
        --n_heads 4 \
        --train_epochs 20 \
        --patience 3 \
        --alpha 0 \
        --batch_size 16 \
        --train_epochs 20 \
        --itr 1 \
        --compare_baseline -1 \
        --learning_rate 0.001
done

for pred_len in 336
do
python -u run.py \
        --is_training 1 \
        --root_path ./dataset/solar/ \
        --data_path solar_AL.csv \
        --model_id solar_96_96_$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --d_layers 1 \
        --factor 3 \
        --enc_in 137 \
        --dec_in 137 \
        --c_out 137 \
        --des 'Exp' \
        --e_layers 2 \
        --d_model 256 \
        --d_ff 256 \
        --dropout 0.1 \
        --n_heads 4 \
        --train_epochs 20 \
        --patience 3 \
        --alpha 0 \
        --batch_size 16 \
        --train_epochs 20 \
        --itr 1 \
        --compare_baseline -1 \
        --learning_rate 0.001
done

for pred_len in 720
do
python -u run.py \
        --is_training 1 \
        --root_path ./dataset/solar/ \
        --data_path solar_AL.csv \
        --model_id solar_96_96_$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --d_layers 1 \
        --factor 3 \
        --enc_in 137 \
        --dec_in 137 \
        --c_out 137 \
        --des 'Exp' \
        --e_layers 2 \
        --d_model 128 \
        --d_ff 256 \
        --dropout 0.1 \
        --n_heads 4 \
        --train_epochs 20 \
        --patience 3 \
        --alpha 0 \
        --batch_size 16 \
        --train_epochs 20 \
        --itr 1 \
        --compare_baseline -1 \
        --learning_rate 0.001
done

#######最好的
#for pred_len in 96 192 336 720
#do
#python -u run.py \
#        --is_training 1 \
#        --root_path ./dataset/solar/ \

#        --data_path solar_AL.csv \
#        --model_id solar_96_96_$pred_len \
#        --model $model_name \
#        --data custom \
#        --features M \
#        --seq_len 96 \
#        --label_len 48 \
#        --pred_len $pred_len \
#        --e_layers 1 \
#        --d_layers 1 \
#        --factor 3 \
#        --enc_in 137 \
#        --dec_in 137 \
#        --c_out 137 \
#        --des 'Exp' \
#        --d_model 256 \
#        --d_ff 512 \
#        --itr 1 \
#        --dropout 0.1 \
#        --train_epochs 20 \
#        --patience 3 \
#        --n_heads 4 \
#        --alpha 0 \
#        --batch_size 16 \
#        --train_epochs 20 \
#        --compare_baseline -1 \
#        --learning_rate 0.001
#done
