from omegaconf import OmegaConf
from hydra.utils import to_absolute_path as abs_path
from os.path import join
import os
from pathlib import Path
import numpy as np
from glob import glob
import tifffile


def get_cfg(cfg, type='track'):
    """
    初始化并配置数据路径，根据运行类型加载相应的图像、掩码和检查点数据。

    Args:
        cfg: 配置对象，包含路径、数据加载器参数和训练参数
        type: 运行类型，可选值为 'train'（训练）、'track'（追踪）、'predict'（预测），默认为 'track'

    Returns:
        配置对象，包含初始化后的所有路径和数据目录
    """
    print('please check the configures:\n')
    print(OmegaConf.to_yaml(cfg))

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    extra_config = OmegaConf.create({
                        'imgs_dir': [],
                        'mask_dir': [],
                        'cp_dir': '',
                        'division_mask': '',
                        'division_mask_dir': [],
                        'flow_dir': [],
                        'result': '',
                        'load': '',
                        'shape': None,
                        'track': {
                            'thr_dist': 10},
                        })
    cfg = OmegaConf.merge(cfg, extra_config)
    cfg.track.thr_dist = cfg.track.max_movenment

    imgs = abs_path(join(cfg.path, 'data', 'img'))
    mask = abs_path(join(cfg.path, 'data', 'mask'))
    cfg.division_mask = abs_path(join(cfg.path, 'data', 'cndd_division'))
    cfg.flow_dir = abs_path(join(cfg.path, 'data', 'flow'))
    cfg.cp_dir = abs_path(join(cfg.path, 'result', 'checkpoint'))
    cfg.result = abs_path(join(cfg.path, 'result'))

    path = [cfg.cp_dir, cfg.division_mask, cfg.flow_dir]
    make_dir(path)

    #  ## please check your img data
    cfg.imgs_dir = np.sort(glob(join(imgs, '*.tif'))).tolist()
    cfg.dataloader.start_frame = min(len(cfg.imgs_dir), cfg.dataloader.start_frame)
    cfg.dataloader.num_frame = min(len(cfg.imgs_dir) - cfg.dataloader.start_frame, cfg.dataloader.num_frame)
    cfg.imgs_dir = cfg.imgs_dir[cfg.dataloader.start_frame: cfg.dataloader.start_frame + cfg.dataloader.num_frame]

    #  ## please check your mask data
    cfg.mask_dir = np.sort(glob(join(mask, '*.tif'))).tolist()[cfg.dataloader.start_frame:
                                   cfg.dataloader.start_frame + cfg.dataloader.num_frame]

    if type == 'train':
        cfg.train.load = abs_path(cfg.train.load)
        img = tifffile.imread(cfg.imgs_dir[0])
        cfg.shape = img.shape
        if img.max() > 1:
            for img in cfg.imgs_dir:
                tifffile.imwrite(img, max_min_morn(tifffile.imread(img)).astype(np.float32))

        cfg.division_mask_dir = np.sort(glob(join(cfg.division_mask, '*.tif'))).tolist()[
            cfg.dataloader.start_frame:
            cfg.dataloader.start_frame + cfg.dataloader.num_frame]
        #  ## please rewrite to fit your divison mask data
        if cfg.dataloader.division_detect:
            # 删除文件夹cfg.division_mask下的文件
            for f in glob(join(cfg.division_mask, '*.tif')):
                os.remove(f)
            from division_detector import main
            main(cfg.imgs_dir, cfg.mask_dir, cfg.division_mask, max_dt=cfg.track.max_movenment, dettype='seg')
            cfg.division_mask_dir = np.sort(glob(join(cfg.division_mask, '*.tif'))).tolist()

    if type == 'track':
        cfg.flow_dir = np.sort(glob(join(cfg.flow_dir, '*.tif'))).tolist()[
                cfg.dataloader.start_frame:
                cfg.dataloader.start_frame + cfg.dataloader.num_frame-1]
        # cfg.shape = tifffile.imread(cfg.flow_dir[0]).shape[1:]
        cfg.train.load = abs_path(cfg.train.load)
        img = tifffile.imread(cfg.imgs_dir[0])
        cfg.shape = img.shape

    if type == 'predict':
        cfg.load = str(np.sort(glob(join(cfg.cp_dir, '*.pth')))[-1])
        [os.remove(f) for f in glob(join(cfg.flow_dir, '*.tif'))]

    return cfg


def max_min_morn(img, max_int=None, min_int=None):
    if max_int is None:
        max_int = img.max()
    if min_int is None:
        min_int = img.min()
    img = (img - min_int) / (max_int - min_int)
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def make_dir(path: list) -> None:
    for p in path:
        save_path = Path(p)
        if not save_path.exists():
            os.makedirs(save_path)
