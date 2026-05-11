import torch
import torch.nn as nn


def warp(x, flo, mode: str = "bilinear"):
    """
    根据光流场对图像/张量进行变形操作，将im2映射回im1的坐标系。

    Args:
        x: 输入图像张量，形状为[B, C, H, W]，表示im2。
        flo: 光流场张量，形状为[B, 2, H, W]，表示像素位移。
        mode: 插值模式，默认为"bilinear"。

    Returns:
        变形后的输出张量，形状与输入x相同。
    """

    flo = torch.cat((flo[:, 1:2], flo[:, 0:1]), dim=1)  ## 不同的库x,y轴不一样， 细胞定位可能是x, y交换的，若出错，请取消这一步
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(device=x.device)

    # grid[:, 0] = grid[:, 0] - flo0[:, 0]
    # grid[:, 1] = grid[:, 1] + flo0[:, 1]
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
    output = nn.functional.grid_sample(x, vgrid, mode=mode, padding_mode='zeros', align_corners=True)

    return output


if __name__ == '__main__':
    # import numpy as np
    # a = torch.ones((2, 1, 10, 10)).cuda()
    # b = np.random.random((2, 2, 10, 10))*10
    # c = warp(a, torch.tensor(b).float().cuda())

    a = torch.ones((1, 1, 2, 2)).cuda()
    b = torch.ones((5, 2, 2, 2)).cuda()
    a[:, :, 0, 0] = 3
    a[:, :, 1, 1] = 2
    a[:, :, 0, 1] = 4
    b[:, 0, 0, 1] = 0
    b[:, 1, 0, 1] = -1
    b[1, 1, 0, 1] = 2
    ee = b.permute(0, 2, 3, 1).reshape((5, -1, 2))

    AA, COUNT = torch.unique(ee, dim=1, return_counts=True)
    print(torch.unique(ee, dim=1, return_counts=True))
    c = warp(a, b)
    d = warp(c, b)
    print(a, b, c, d)

