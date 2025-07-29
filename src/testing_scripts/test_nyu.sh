
ckpt=/home/descfly/Projects/NLSPN_ECCV20-master/results/NLSPN_NYU.pt

# for sample in 1 5 50 100 200 300 400 500 
# for sample in 1 50 100 200 500 
for sample in 500
# for sample in 20 2000 10000

# do
# python main.py --dir_data /home/descfly/data/nyudepthv2 --data_name NYU --split_json ../data_json/nyu.json \
#     --gpus 0 --max_depth 10.0 --num_sample $sample \
#     --test_only --pretrain $ckpt \
#     --log_dir /data/compare/metric/NLSPN/experiments/ \
#     --preserve_input \
#     --patch_height 228 --patch_width 304\
#     --save "test_nyu_8msk_sample${sample}" \
#     --legacy \
#     --save_result_only  
#     # --save "$test_nyu_sample${sample}" \
#     # --save 'nyu_1.10' \
#     # --save_result_only  
#     # --save_full --save_image  
#     # --save_full --save_pointcloud_visualization
# done

do
python main_memory.py --dir_data /home/descfly/data/nyudepthv2 --data_name NYU --split_json ../data_json/nyu.json \
    --gpus 0 --max_depth 10.0 --num_sample $sample \
    --test_only --pretrain $ckpt \
    --log_dir /data/compare/metric/NLSPN/experiments/ \
    --preserve_input --batch_size 1 \
    --save "test_nyu_sample${sample}_inference" \
    --legacy \
    # --save_result_only  
    # --save "$test_nyu_sample${sample}" \
    # --save 'nyu_1.10' \
    # --save_result_only  
    # --save_full --save_image  
    # --save_full --save_pointcloud_visualization
done