GRU_iters=1
test_augment=0
optim_layer_input_clamp=1.0
depth_activation_format='exp'

ckpt=/home/descfly/Projects/NLSPN_ECCV20-master/results/NLSPN_NYU.pt

for sample in 1500 500 150
do
  python main.py --dir_data /home/descfly/data/void_release/void_${sample} --data_name VOID \
    --gpus 0 --max_depth 5.0 --num_sample $sample \
    --test_only --pretrain $ckpt \
    --log_dir /data/compare/metric/NLSPN/experiments/ \
    --preserve_input --batch_size 1 \
    --save "test_nyu_sample${sample}_inference" \
    --legacy 
done


