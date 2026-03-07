
ckpt=../checkpoints/NLSPN_NYU.pt
save_idx=318
num_masks=100
# for sample in 1 5 50 100 200 300 400 500 
# for sample in 1 50 100 200 500 
#for sample in 5 10 20
# for sample in 20 2000 10000

#do
#python main.py --dir_data /home/descfly/data/nyudepthv2 --data_name NYU --split_json ../data_json/nyu.json \
#    --gpus 0 --max_depth 10.0 --num_sample $sample \
#    --test_only --pretrain $ckpt \
#    --log_dir /data/compare/metric/NLSPN/experiments/ \
#    --preserve_input \
#    --save "test_nyu_sample${sample}_mask8" \
#    --legacy \
#    --save_result_only
#    # --patch_height 228 --patch_width 304\
#    # --save "$test_nyu_sample${sample}" \
#    # --save 'nyu_1.10' \
#    # --save_result_only
#    # --save_full --save_image
#    # --save_full --save_pointcloud_visualization
#done
#for sample in 50 100 200 500 1000 2000 5000 10000 20000
#for sample in 10
for sample in 1000 5000 10000
do
    python main.py --dir_data ../datas/nyudepthv2 --data_name NYU  --split_json ../data_json/nyu.json \
    --patch_height 228 --patch_width 304 --gpus 0 --max_depth 10.0 --num_sample $sample \
    --test_only --pretrain $ckpt --preserve_input --legacy \
    --log_dir ../experiments/masks${num_masks}p${sample}/ --save_idx ${save_idx} \
    --num_masks ${num_masks}
#    --save_single
done

# do
# python main_memory.py --dir_data /home/descfly/data/nyudepthv2 --data_name NYU --split_json ../data_json/nyu.json \
#     --gpus 0 --max_depth 10.0 --num_sample $sample \
#     --test_only --pretrain $ckpt \
#     --log_dir /data/compare/metric/NLSPN/experiments/ \
#     --preserve_input --batch_size 1 \
#     --save "test_nyu_sample${sample}_inference" \
#     --legacy \
#     # --save_result_only  
#     # --save "$test_nyu_sample${sample}" \
#     # --save 'nyu_1.10' \
#     # --save_result_only  
#     # --save_full --save_image  
#     # --save_full --save_pointcloud_visualization
# done