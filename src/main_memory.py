"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    main script for training and testing.
"""


from config import args as args_config
import time
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = args_config.port

import json
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import utility
from model import get as get_model
from data import get as get_data
from loss import get as get_loss
from metric import get as get_metric
from summary import get as get_summary
from metric import getNew as get_metric16bit
from summary import getNew as get_summary16bit

# Multi-GPU and Mixed precision supports
# NOTE : Only 1 process per GPU is supported now
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import pandas as pd
import torch.utils.benchmark as benchmark
# Minimize randomness
torch.manual_seed(args_config.seed)
np.random.seed(args_config.seed)
random.seed(args_config.seed)
torch.cuda.manual_seed_all(args_config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def check_args(args):
    if args.batch_size < args.num_gpus:
        print("batch_size changed : {} -> {}".format(args.batch_size,
                                                     args.num_gpus))
        args.batch_size = args.num_gpus

    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)

            new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume

    return new_args   

def test(args):
    # Prepare dataset
    data = get_data(args)
    data_test = data(args, 'test')
    loader_test = DataLoader(dataset=data_test, batch_size=1,
                             shuffle=False, num_workers=args.num_threads)

    # Network
    model = get_model(args)
    net = model(args)
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), "file not found: {}".format(args.pretrain)
        f = open(args.pretrain, 'rb')
        checkpoint = torch.load(f, encoding='latin1')
        key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)
        if key_u:
            print('Unexpected keys :')
            print(key_u)
        if key_m:
            print('Missing keys :')
            print(key_m)
            raise KeyError

    net = nn.DataParallel(net)

    metric16bit = get_metric16bit(args)
    metric16bit = metric16bit(args)
    summary16bit = get_summary16bit(args)

    try:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir + '/test', exist_ok=True)
    except OSError:
        pass

    writer_test_new = summary16bit(args.save_dir, 'test', args, None, metric16bit.metric_name)

    net.eval()

    num_sample = len(loader_test) * loader_test.batch_size

    pbar = tqdm(total=num_sample)

    # 新增 torch.utils.benchmark.Timer 相关代码
    # 创建 Timer 计时器，命令字符串是 net(sample) ，
    # 但这里 sample 需要是准备好的CUDA tensor，我们在循环里动态赋值，所以先占位
    # 这里我们改用循环内创建Timer（更准确），或者用 timeit.Timer 动态测试

    t_total = 0.0
    times = []

    for batch, sample in enumerate(loader_test):
        sample = {key: val.cuda() for key, val in sample.items() if val is not None}
        torch.cuda.reset_max_memory_allocated(device=0)
        timer = benchmark.Timer(
            stmt='net(sample)',
            globals={'net': net, 'sample': sample}
        )
        
        measured_time = timer.timeit(1000).mean
        print(f"{measured_time} sec")
        break
    peak_memory = torch.cuda.max_memory_allocated(device=0) / 1024 ** 3
    print(f"[Batch {batch}] Inference time: {measured_time:.6f}s | Peak memory: {peak_memory:.3f} GB")
    model = net.module if isinstance(net, torch.nn.DataParallel) else net
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")


    
def main(args):
    if not args.test_only:
        if args.no_multiprocessing:
            train(0, args)
        else:
            assert args.num_gpus > 0

            spawn_context = mp.spawn(train, nprocs=args.num_gpus, args=(args,),
                                     join=False)

            while not spawn_context.join():
                pass

            for process in spawn_context.processes:
                if process.is_alive():
                    process.terminate()
                process.join()

            args.pretrain = '{}/model_{:05d}.pt'.format(args.save_dir,
                                                        args.epochs)

    test(args)


if __name__ == '__main__':
    args_main = check_args(args_config)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args_main)):
        print(key, ':',  getattr(args_main, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')

    main(args_main)
