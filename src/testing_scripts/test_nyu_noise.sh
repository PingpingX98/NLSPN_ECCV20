
ckpt=/home/descfly/Projects/NLSPN_ECCV20-master/results/NLSPN_NYU.pt
# for sample in 5 50 100 200 300 400 500 
# for sample in 1 5 50 100 200 500 1000 5000 20000
# for sample in 1 5 50 100
# for sample in 300 400
for noise_type in gaussian impulse rayleigh gamma exponential uniform
do
python main.py --dir_data /home/descfly/data/nyudepthv2 --data_name NYU --split_json ../data_json/nyu.json \
    --gpus 0 --max_depth 10.0 --num_sample 500 \
    --test_only --pretrain $ckpt \
    --log_dir /data/compare/metric/NLSPN/experiments/ \
    --preserve_input --batch_size 1 \
    --save "test_nyu_noise${noise_type}" \
    --add_noise --noise_type ${noise_type} \
    --legacy \
    # --save_result_only  
    # --save "$test_nyu_sample${sample}" \
    # --save 'nyu_1.10' \
    # --save_result_only  
    # --save_full --save_image  
    # --save_full --save_pointcloud_visualization
done

# do
# python main_time.py --dir_data /home/descfly/data/nyudepthv2 --data_name NYU --split_json ../data_json/nyu.json \
#     --gpus 0 --max_depth 10.0 --num_sample $sample --batch_size 1 \
#     --GRU_iters $GRU_iters --optim_layer_input_clamp $optim_layer_input_clamp \
#     --depth_activation_format $depth_activation_format \
#     --test_only --test_augment $test_augment --pretrain $ckpt \
#     --log_dir /data/compare/metric/OGNI-DC/experiments/ \
#     --save "test_nyu_sample${sample}_inference" \
#     # --save_result_only
#     # --save 'nyu_1.10' \
# done
