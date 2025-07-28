
ckpt=/home/descfly/Projects/NLSPN_ECCV20-master/results/NLSPN_KITTI_DC.pt

for lidar_lines in 64
do
  python main_time.py --dir_data /home/descfly/data/kitti_depth \
  --data_name KITTIDC --split_json ../data_json/kitti_dc.json \
  --patch_height 352 --patch_width 1216 --gpus 0 --max_depth 90.0 --top_crop 0\
  --num_sample 0 --test_only --pretrain $ckpt \
  --preserve_input --legacy  \
  --lidar_lines $lidar_lines \
  --log_dir /data/compare/metric/NLSPN/experiments/ \
  --save "val_kitti_lines${lidar_lines}" \
  --save_result_only
  done
