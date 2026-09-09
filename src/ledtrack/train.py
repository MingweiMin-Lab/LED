import logging
import os
import sys
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
from .data_loader import Train_Dataset
from torch.utils.data import DataLoader
import hydra
from hydra.utils import to_absolute_path as abs_path
import warnings
warnings.filterwarnings("ignore")
from .loss import Loss
from .models import UNet
from .utils import get_cfg
from omegaconf import DictConfig


def train_net(net,
              device,
              cfg,
              ):
    """
    训练神经网络模型的主流程。

    Args:
        net: 神经网络模型实例
        device: 计算设备 (torch.device，CPU或CUDA)
        cfg: 配置对象，包含训练参数、数据路径等设置
    """
    train = Train_Dataset(cfg)
    n_train = len(train)

    epochs = cfg.train.epochs
    batch_size = cfg.train.batch_size
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    # writer = SummaryWriter(log_dir=abs_path('./logs'), flush_secs=120, comment=f'LR_{cfg.train.lr}_BS_{batch_size}')
    global_step = 0

    loss_w = Loss()
    optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {cfg.train.lr}
        Training size:   {len(train)}
        Checkpoints:     {cfg.cp_dir}
        Device:          {device.type}
        Interval:        {cfg.dataloader.itv}
        Optimizer:       {optimizer.__class__.__name__}
    ''')

    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                net.train()

                imgs = batch['img'].to(device=device, dtype=torch.float32)
                cell_prob = batch['cell_prob'].to(device=device, dtype=torch.float32)
                mask_nxt = batch['mask_nxt'].to(device=device, dtype=torch.float32)
                division_mask = batch['division_mask'].to(device=device, dtype=torch.float32)

                flows_pred, weight = net(imgs)
                loss = loss_w(flows_pred, imgs, mask_nxt, division_mask, cell_prob, weight)

                # writer.add_scalar('Loss/train', loss.item(), global_step)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.)
                optimizer.step()
                pbar.update(n=imgs.shape[0])
                epoch_loss += loss.item()
                global_step += 1

            pbar.set_postfix(**{'loss (batch)': epoch_loss/n_train*batch_size})
        scheduler.step()
        torch.save(net.state_dict(),
                   abs_path(os.path.join(cfg.cp_dir, f'CP_epoch{str(epoch + 1).zfill(2)}.pth')))
        # logging.info(f'Checkpoint {epoch + 1} saved !')
    torch.save(net.state_dict(), 'CP.pth')

    # writer.close()


@hydra.main(config_path=abs_path(r'config'), version_base='1.3', config_name='tracker')
def train_model(cfg: DictConfig):
    cfg = get_cfg(cfg, type='train')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=2, bilinear=True)

    # # 检查可用的GPU数量
    # if torch.cuda.device_count() > 1:
    #     print(f"Using device: {torch.cuda.device_count()} GPUs")
    #     # 使用所有GPU
    #     net = nn.DataParallel(net).to(device=device)
    # else:
    #     net.to(device=device)
    net.to(device=device)
    if cfg.train.train_load:
        net.load_state_dict(
            torch.load(cfg.train.load, map_location=device)
        )
        logging.info(f'Model loaded from {cfg.train.load}')

    # # 检查可用的GPU数量
    # if torch.cuda.device_count() > 1:
    #     print(f"Using device: {torch.cuda.device_count()} GPUs")
    #     # 使用所有GPU
    #     net = nn.DataParallel(net).cuda()
    # else:
    #     net.to(device=device)

    try:
        train_net(net=net,
                  device=device,
                  cfg=cfg)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), r'result\interrupted.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == '__main__':

    train_model()

