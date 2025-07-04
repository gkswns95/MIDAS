CUDA_VISIBLE_DEVICES=0 \
python train.py \
--trial 9999 \
--dataset afootball \
--model imputeformer \
--missing_pattern playerwise \
--missing_rate 0.5 \
--player_order xy_sort \
--team_size 6 \
--single_team \
--n_features 2 \
--window_size 50 \
--window_stride 5 \
--n_epochs 100 \
--start_lr 1e-3 \
--min_lr 1e-5 \
--batch_size 1 \
--print_every_batch 50 \
--save_every_epoch 50 \
--seed 100 \
--cuda \
--n_nodes 22 \
--input_embedding_dim 32 \
--feed_forward_dim 256 \
--learnable_embedding_dim 96 \
--n_temporal_heads 4 \
--n_layers 3 \
--proj_dim 50 \
--dropout 0 \