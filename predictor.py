import numpy as np
import torch.nn as nn
from tqdm import tqdm
import tifffile
import torch
import os
import logging
import sys
from models import UNet
import hydra
from utils import get_cfg
from torch.utils.data import DataLoader
from data_loader import Train_Dataset
from os.path import join
from hydra.utils import to_absolute_path as abs_path
from omegaconf import DictConfig
import datetime
now = datetime.datetime.now()
# from warp import warp


def stitch_tiles(imgs, ol):
    """stitch tiles
    Args:
        imgs:(im1 im3
              im2 im4)

    """

    tile_num = len(imgs) ** 0.5
    if tile_num == 1:
        return imgs[0]

    _, w, h = imgs[0].shape
    overlap = ol * 2
    img_stitch = np.zeros((np.array((2, (w - ol) * tile_num, (h - ol) * tile_num))).astype(np.uint16))
    overlap_weight = np.linspace(0, 1, overlap).reshape((1, -1)).repeat(2, axis=0)
    # weight = np.zeros_like(imgs[0])
    if tile_num == 2:

        imgs[0][:, -overlap:] = imgs[0][:, -overlap:] * overlap_weight.reshape(2, -1, 1)[:, ::-1]
        imgs[0][:, :, -overlap:] = imgs[0][:, :, -overlap:] * overlap_weight.reshape(2, 1, -1)[:, :, ::-1]
        img_stitch[:, :w, :h] = imgs[0]

        img = imgs[1]
        img[:, -overlap:] = img[:, -overlap:] * overlap_weight.reshape(2, -1, 1)[:, ::-1]
        img[:, :, :overlap] = img[:, :, :overlap] * overlap_weight.reshape(2, 1, -1)
        img_stitch[:, :w, -h:] = img_stitch[:, :w, -h:] + img

        img = imgs[2]
        img[:, :overlap] = img[:, :overlap] * overlap_weight.reshape(2, -1, 1)
        img[:, :, -overlap:] = img[:, :, -overlap:] * overlap_weight.reshape(2, 1, -1)[:, :, ::-1]
        img_stitch[:, -w:, :h] = img_stitch[:, -w:, :h] + img

        imgs[3][:, :overlap] = imgs[3][:, :overlap] * overlap_weight.reshape(2, -1, 1)
        imgs[3][:, :, :overlap] = imgs[3][:, :, :overlap] * overlap_weight.reshape(2, 1, -1)
        img_stitch[:, -w:, -h:] = img_stitch[:, -w:, -h:] + imgs[3]

    else:
        ValueError: 'unsupported tiles!'

    return img_stitch


def pred_net(net, loader, device, cfg, start=1):
    """Evaluation
    Args:
        net (nn.Module): model
        loader (DataLoader): data loader
        device (torch.device): device

    """
    net.eval()
    n_val = len(loader)  # the number of batch
    # tanhs = nn.Tanhshrink()
    tiles = []
    i = start
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:

            imgs = batch['img'].to(device=device, dtype=torch.float32)

            with torch.no_grad():
                flows_pred, _ = net(imgs)

            if cfg.dataloader.if_crop:
                tiles.append(flows_pred[0].detach().cpu().numpy())
                if len(tiles) == cfg.dataloader.pred_tile_num:
                    flows = stitch_tiles(tiles, cfg.dataloader.overlap)

                    path = join(abs_path(cfg.flow_dir), now.strftime('%Y-%m-%d-') + str(i).zfill(4) + '.tif')
                    tifffile.imwrite(path, flows)
                    i = i + 1
                    tiles = []
            else:
                path = join(abs_path(cfg.flow_dir), now.strftime('%Y-%m-%d-') + str(i).zfill(4) + '.tif')
                tifffile.imwrite(path, flows_pred[0].detach().cpu().numpy())
                i = i + 1

            pbar.update()


@hydra.main(config_path=abs_path('config'), version_base='1.3', config_name='tracker')
def predictor(cfg: DictConfig):
    cfg = get_cfg(cfg, type='predict')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=2, bilinear=True)

    net.to(device=device)
    # net = nn.DataParallel(net).to(device=device)
    # # 检查可用的GPU数量
    # if torch.cuda.device_count() > 1:
    #     print(f"Using device: {torch.cuda.device_count()} GPUs")
    #     # 使用所有GPU
    #     net = nn.DataParallel(net).to(device=device)
    # else:
    #     net.to(device=device)

    net.load_state_dict(
        torch.load(cfg.load, map_location=device)
    )
    logging.info(f'Model loaded from {cfg.load}')

    cfg.train.val = True
    # cfg.dataloader.if_crop = False
    # cfg.dataloader.tile_num = 1
    dataset = Train_Dataset(cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    try:
        pred_net(net=net,
                 loader=loader,
                 device=device,
                 cfg=cfg,
                 start=cfg.dataloader.start_frame
                 )

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':

    imgs = [np.ones((2, 20, 15)) for _ in range(4)]

    img_stitch = stitch_tiles(imgs, ol=4)

    predictor()
