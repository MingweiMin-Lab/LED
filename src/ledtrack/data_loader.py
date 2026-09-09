import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from tifffile import imread
from skimage import morphology
from skimage.morphology import square, opening


class Train_Dataset(Dataset):
    def __init__(self, cfg):

        self.imgs_dir = cfg.imgs_dir
        self.masks_dir = cfg.mask_dir
        self.division_mask = cfg.division_mask_dir
        self.division_detect = cfg.dataloader.division_detect
        self.itv = cfg.dataloader.itv
        self.if_crop = cfg.dataloader.if_crop
        self.val = cfg.train.val
        if self.if_crop:
            if self.val:
                self.tile_num = cfg.dataloader.pred_tile_num
            else:
                self.tile_num = cfg.dataloader.tile_num
        else:
            self.tile_num = 1
        # self.height, self.width = cfg.dataloader.height, cfg.dataloader.width
        # self.division_detect = cfg.division_detect
        #
        # self.Ellip_log = Ellip_log_blob()

        logging.info(f'Creating dataset with {len(self.imgs_dir)} examples')

    #Data validity check
    def __check_validity__(self, dirs) -> list:
        legal_idx = []
        for i, path in enumerate(dirs):
            m = imread(path)
            if len(np.unique(m)) > 1:
                legal_idx.append(i)
            # assert len(np.unique(m)) > 1, f'illegal idx:{i}th image has no cell'
        return legal_idx

    def crop(self, img, ith_tile, rows=None, cols=None, overlap=32):
        assert self.tile_num > ith_tile
        if rows is None and cols is None:
            if self.tile_num == 1:
                return img
            if self.tile_num == 4:
                rows, cols = 2, 2
            elif self.tile_num == 9:
                rows, cols = 3, 3
            elif self.tile_num == 16:
                rows, cols = 4, 4
            elif self.tile_num == 25:
                rows, cols = 5, 5
            else:
                raise ValueError('unsupported tile_num!')

        tile_w, tile_h = img.shape[-2] // rows + overlap, img.shape[-1] // cols + overlap
        if tile_w % 2 != 0:
            tile_w += 1
        if tile_h % 2 != 0:
            tile_h += 1

        ith_row, ith_col = ith_tile // rows, ith_tile % cols

        x_start, x_end = int(ith_row * tile_w) - overlap // 2, int((ith_row + 1) * tile_w) - overlap // 2
        y_start, y_end = int(ith_col * tile_h) - overlap // 2, int((ith_col + 1) * tile_h) - overlap // 2

        if x_start < 0:
            x_start, x_end = 0, tile_w
        if y_start < 0:
            y_start, y_end = 0, tile_h

        if ith_row == rows - 1:
            x_start, x_end = img.shape[-2] - tile_w,  img.shape[-2]
        if ith_col == cols - 1:
            y_start, y_end = img.shape[-1] - tile_h, img.shape[-1]

        if len(img.shape) == 3:
            tile = img[:, x_start:x_end, y_start:y_end]
        else:
            tile = img[x_start:x_end, y_start:y_end]

        return tile

    def max_min_morn(self, img, max_int=None, min_int=None):
        if max_int is None:
            max_int = img.max()
        if min_int is None:
            min_int = img.min()
        img = (img - max_int) / (max_int - min_int)
        img[img > 1] = 1
        img[img < 0] = 0
        return img

    def __len__(self):
        if self.if_crop:
            return self.tile_num * (len(self.imgs_dir) - self.itv)
        return len(self.imgs_dir) - self.itv

    def __getitem__(self, i):
        idx = i // self.tile_num
        ith_tile = i % self.tile_num

        img1_file = self.imgs_dir[idx]
        img2_file = self.imgs_dir[idx + self.itv]
        
        img1, img2 = imread(img1_file), imread(img2_file)
        # if len(img1.shape) > 2:
        #     img1, img2 = img1[0], img2[0]

        if img1.max() > 1:
            img1 = self.max_min_morn(img1)
            img2 = self.max_min_morn(img2)

        if self.if_crop:
            img1 = self.crop(img1, ith_tile)
            img2 = self.crop(img2, ith_tile)

        im_diff = img2 - img1
        im_nxt = im_diff > 0

        im_nxt = opening(im_nxt, square(3))
        im_nxt = morphology.remove_small_holes(im_nxt, area_threshold=100, connectivity=1)
        im_nxt = morphology.remove_small_objects(im_nxt, min_size=80, connectivity=1)

        assert img1.shape == img2.shape, \
            f'Images are not the same size, {img1.size} and {img2.size}'

        if len(img1.shape) <= 2:
            img1 = np.expand_dims(img1, axis=0)
            img2 = np.expand_dims(img2, axis=0)
        img = np.concatenate([img1, img2, im_nxt * img2], axis=0)

        if self.val:
            return {
                'img': torch.from_numpy(img).type(torch.FloatTensor),
            }
        else:
            if self.division_detect:
                division_mask = imread(self.division_mask[idx])
                # division_mask = imread(self.division_mask[idx + self.itv])
                division_mask = self.crop(division_mask, ith_tile)
            else:
                division_mask = np.zeros_like(img[0:1])

            mask0_file = self.masks_dir[idx]
            mask1_file = self.masks_dir[idx + self.itv]
            mask0 = imread(mask0_file)
            mask1 = imread(mask1_file)
            # mask0 = np.array(open(mask0_file))
            # mask1 = np.array(open(mask1_file))

            if self.if_crop:
                mask0 = self.crop(mask0, ith_tile)
                mask1 = self.crop(mask1, ith_tile)

            labels = np.unique(mask0)[1:]
            centroid = []
            cell_prob = np.zeros(img[0].shape, dtype=np.uint8)
            for label in labels:
                centroid.append(np.mean(np.array(np.where(mask0 == label)).T, axis=0))
            centroid = np.array(centroid).astype(np.int16).T
            cell_prob[centroid[0], centroid[1]] = 1

            img = torch.from_numpy(img).type(torch.FloatTensor)
            cell_prob = torch.from_numpy(cell_prob).type(torch.FloatTensor)
            division_mask = torch.from_numpy(division_mask).type(torch.FloatTensor)
            im_nxt_m = torch.from_numpy(mask1.astype(np.int16)).type(torch.FloatTensor).unsqueeze(dim=0)

            return {
                'img': img,
                'cell_prob': cell_prob,
                'division_mask': division_mask,
                'mask_nxt': im_nxt_m,
            }


