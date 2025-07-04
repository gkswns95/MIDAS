CUDA_VISIBLE_DEVICES=0 \
python train.py \
--trial 511 \
--dataset basketball \
--model imputeformer \
--missing_pattern uniform \
--missing_rate 0.5 \
--normalize \
--flip_pitch \
--player_order xy_sort \
--team_size 5 \
--n_features 2 \
--window_size 200 \
--window_stride 5 \
--n_epochs 100 \
--start_lr 1e-3 \
--min_lr 1e-5 \
--batch_size 32 \
--print_every_batch 50 \
--save_every_epoch 50 \
--seed 100 \
--cuda \
--n_nodes 10 \
--input_embedding_dim 32 \
--feed_forward_dim 256 \
--learnable_embedding_dim 96 \
--n_temporal_heads 4 \
--n_layers 3 \
--proj_dim 50 \
--dropout 0 \