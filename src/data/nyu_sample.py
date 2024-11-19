import torch

def uniform_sample1(dep, num_sample):
    channel, height, width = dep.shape

    assert channel == 1

    idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

    num_idx = len(idx_nnz)
    #print(f"the lenth of idx_nn is {num_idx}")
    if num_idx == 0:
        return torch.zeros_like(dep)

    idx_sample = torch.linspace(0, num_idx-1, num_sample, dtype=int)
    #print(f"the idx_sample is {len(idx_sample)}")
    idx_nnz = idx_nnz[idx_sample[:]]

    mask = torch.zeros((channel*height*width))
    mask[idx_nnz] = 1.0
    mask = mask.view((channel, height, width))

    dep_sp = dep * mask.type_as(dep)

    return dep_sp

def uniform_sample2(dep, num_sample):
    channel, height, width = dep.shape
    
    assert channel == 1

    idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

    num_idx = len(idx_nnz)
    # uniform sampling
    if num_idx == 0:
        return torch.zeros_like(dep)
        
    interval = max (1, num_idx // num_sample)   # uniform interbal
    idx_sample = idx_nnz[::interval][:num_sample]   
    #print(f"the idx_sample is {idx_sample}")
    #print(f"the lenth of idx_sample is {len(idx_sample)}")
    idx_nnz = idx_nnz[idx_sample[:]]

    mask = torch.zeros((channel*height*width))
    mask[idx_nnz] = 1.0
    mask = mask.view((channel, height, width))

    dep_sp = dep * mask.type_as(dep)

    return dep_sp


def uniform_sample3(dep, num_sample):
    channel, height, width = dep.shape
    assert channel == 1

    # 横向均匀采样25个点，竖向均匀采样20个点
    row_idx = torch.linspace(0, height - 1, num_sample // 25, dtype=torch.long)  # 竖向20个点
    col_idx = torch.linspace(0, width - 1, num_sample // 20, dtype=torch.long)   # 横向25个点

    # 生成所有采样点的二维网格
    row_grid, col_grid = torch.meshgrid(row_idx, col_idx, indexing="ij")
    idx = torch.stack([row_grid.flatten(), col_grid.flatten()], dim=1)

    # 构建掩码，标记采样点的位置
    mask = torch.zeros((height, width), dtype=torch.float32)
    mask[idx[:, 0], idx[:, 1]] = 1.0

    dep_sp = dep * mask.unsqueeze(0)  # 扩展 mask 匹配 dep 的维度

    return dep_sp
