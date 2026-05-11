import torch
from torch.autograd import Variable
from math import exp
import torch.nn as nn
import torch.nn.functional as F
from warp import warp


class Loss(nn.Module):
    def __init__(self,
                 log_var=True,
                 dynamic_weight=True,
                 weight=[1, 1, 1, 1, 1, 1, 1, 1]):
        super().__init__()
        self.photo_loss = PhotoLoss()
        self.mse = nn.MSELoss()
        self.ssim_loss = SSIM()

        self.cluster = ClusterLoss()
        self.cos_loss = CosLoss()
        self.morph_sim_loss = MorphSimLoss()
        self.div_loss = DivisionLoss()
        self.collision_loss = CollisionLoss()

        self.log_var_weight = log_var
        self.dynamic_weight = dynamic_weight

        self.w = torch.tensor(weight)  # weight是各个loss的权重

    def forward(self, flows_pred, imgs, mask_nxt, division_mask, cell_prob, w):
        """
        计算细胞追踪的多任务损失函数

        Args:
            flows_pred: 预测的光流场
            imgs: 输入图像序列
            mask_nxt: 下一帧的细胞掩码
            division_mask: 细胞分裂掩码
            cell_prob: 细胞概率图
            w: 损失权重参数

        Returns:
            加权后的总损失值
        """
        output = warp(imgs[:, :1], flows_pred)
        losses = torch.stack((
            # # ##### 相似度計算
            self.photo_loss(output, imgs[:, 1:2]),
            self.mse(output, imgs[:, 1:2]),
            self.ssim_loss(output, imgs[:, 1:2]),

            # # # # # ## 正则化
            self.cluster(flows_pred, cell_prob, mask_nxt, output, imgs[:, 1:2]),
            self.morph_sim_loss(flows_pred, mask_nxt),
            self.cos_loss(flows_pred),
            self.div_loss(flows_pred, division_mask),
            self.collision_loss(flows_pred)
        ))
        if self.dynamic_weight:
            dynamic_weight = losses.detach() / (losses.detach().sum() + 1e-6)
        else:
            dynamic_weight = 1

        if self.log_var_weight:
            loss = 0.5 * torch.exp(-w[:len(losses)]) * dynamic_weight * losses + w[:len(losses)]
        else:
            loss = self.w[:len(losses)] * dynamic_weight * losses

        assert not torch.isnan(loss.sum()), 'loss is nan! please check your data or the gradian!'

        return loss.sum()


class CollisionLoss(nn.Module):
    def __init__(self, sample_size=50):
        super().__init__()
        self.sample_size = sample_size
        # self.KLD = KLDensityLoss()

    def get_collision(self, displaced_points, mask=None):
        x = displaced_points.unsqueeze(1)  # (N,1,D)
        y = displaced_points.unsqueeze(0)  # (1,N,D)
        dist_matric = torch.cdist(x, y).squeeze()  # (N,N)
        dist = dist_matric + 1e6 * torch.eye(displaced_points.shape[0]).to(dist_matric.device)
        dist = dist[dist < 1].view(-1)

        col_loss = torch.mean(1 - dist) if len(dist) > 0 else 0
        return col_loss

    def forward(self, flows_pred):
        """
        计算光流的冲突损失

        对预测的光流场进行反向计算，检测每个像素的目标位置是否发生冲突。
        如果多个像素映射到同一位置，则视为冲突。

        Args:
            flows_pred: 预测的光流场，shape为(B, 2, H, W)，其中B为batch大小，
                       2个通道分别表示x和y方向的位移

        Returns:
            torch.Tensor: 所有样本的冲突损失总和
        """
        if flows_pred.shape[2] > self.sample_size:
            x = torch.randint(0, flows_pred.shape[2] - self.sample_size, (flows_pred.shape[0], 1))[:, 0]
            flows_pred = torch.stack([flows_pred[b, :, xb: xb + self.sample_size] for b, xb in enumerate(x)])
        if flows_pred.shape[3] > self.sample_size:
            y = torch.randint(0, flows_pred.shape[3] - self.sample_size, (flows_pred.shape[0], 1))
            flows_pred = torch.stack([flows_pred[b, :, :, yb: yb + self.sample_size] for b, yb in enumerate(y)])

        ## 不同的库x,y轴不一样， 可能x, y交换
        flo = torch.cat((flows_pred[:, 1:2], flows_pred[:, 0:1]), dim=1)
        B, _, H, W = flows_pred.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if flows_pred.is_cuda:
            grid = grid.to(device=flows_pred.device)
        vgrid = grid + flo

        vgrid_pos = vgrid.permute(0, 2, 3, 1).view(B, -1, 2)
        collision = torch.stack([self.get_collision(vgrid_pos[b]) for b in range(B)])

        return torch.sum(collision)


class DivisionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y, mask):

        as_loss = torch.zeros(1, requires_grad=True).to(y.device)
        weight0 = torch.unique(mask)
        weight = weight0[weight0 > 0]

        for w in weight:
            orrd1 = torch.where(torch.tensor(mask == w))
            orrd2 = torch.where(torch.tensor(mask == -w))

            # one of daughter cells might be cropped
            if len(orrd2[0]) < 1 or len(orrd1[0]) < 1:
                continue

            divx1 = y[:, 0][orrd1].mean()
            divy1 = y[:, 1][orrd1].mean()
            divx2 = y[:, 0][orrd2].mean()
            divy2 = y[:, 1][orrd2].mean()
            
            mag1 = torch.norm(torch.stack((divx1, divy1)), 2, dim=0)
            mag2 = torch.norm(torch.stack((divx2, divy2)), 2, dim=0)
            cos_sim = (1 + (divx1 * divx2 + divy1 * divy2) /
                       torch.max(mag1 * mag2, torch.tensor(1e-8)))
            as_loss += w * (cos_sim + torch.pow(mag1 - mag2, 2))

        return as_loss[0] / max(len(weight), 1)

class ClusterLoss(nn.Module):
    '''
    cluster loss
    '''
    def __init__(self, error='relative', p='l2'):
        super().__init__()
        self.p = p
        self.error = error
    # # # cell_based
    def forward(self, flow, cell_prob, mask, im2, wrap_im2):

        batch, _, w, h = flow.shape

        sim_error = (im2 - wrap_im2).pow(2)
        # sim_error = torch.abs(im2 - wrap_im2)

        loss = torch.zeros(1, requires_grad=True).to(flow.device)[0]
        # loss = torch.exp(-torch.abs(flow * mask.repeat(1, 2, 1, 1))).mean()
        for b in range(batch):
            centx, centy = torch.where(cell_prob[b] > 0)
            index = torch.unique(mask[b, 0])[1:]
            cluster_loss = torch.zeros(1, requires_grad=True).to(flow.device)
            for idx in index:
                cell = mask[b, 0] == idx
                if self.error == 'absolute':
                    sim_weight = sim_error[b, 0][cell].sum()
                elif self.error == 'relative':
                    sim_weight = sim_error[b, 0][cell].sum() / max(im2[b, 0][cell].sum(), 1)
                else:
                    sim_weight = torch.ones(1, requires_grad=True).to(flow.device)

                xx, yy = torch.where(torch.tensor(cell))
                vx = flow[b, 0, xx, yy].mean()
                vy = flow[b, 1, xx, yy].mean()
                x = vx + torch.mean(xx.to(torch.float32))
                y = vy + torch.mean(yy.to(torch.float32))
                if self.p == 'l2':
                    mag = torch.pow(x - centx, 2) + torch.pow(y - centy, 2)
                    # mag = torch.norm(torch.stack((x - centx, y - centy)), 2, dim=0)
                else:
                    mag = torch.abs(x - centx) + torch.abs(y - centy)

                cluster_loss += sim_weight * torch.min(mag)

            loss += cluster_loss[0] / max(len(index), 1)

        return loss / batch


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

        self.window = window
        self.channel = channel

        return 1 - self.ssim(img1, img2)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def ssim(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class CosLoss(nn.Module):
    """
    win-based (windows size = (3, 3)) CosineSimilarity loss
    """
    def __init__(self, c=4):
        super(CosLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.pad = nn.ConstantPad2d(1, 0)
        self.connect = c

    def forward(self, pred):

        pred_pad = self.pad(pred)

        if self.connect == 4:
            cos_sim = (
                        self.cos(pred, pred_pad[:, :, 1:-1, :-2]) +
                        self.cos(pred, pred_pad[:, :, :-2, 1:-1]) +
                        self.cos(pred, pred_pad[:, :, 1:-1, 2:]) +
                        self.cos(pred, pred_pad[:, :, 2:, 1:-1])
                      )
            return 4 - cos_sim.mean()
        elif self.connect == 8:
            cos_sim = (
                    self.cos(pred, pred_pad[:, :, 2:, 2:]) +
                    self.cos(pred, pred_pad[:, :, :-2, :-2]) +
                    self.cos(pred, pred_pad[:, :, 1:-1, :-2]) +
                    self.cos(pred, pred_pad[:, :, :-2, 1:-1]) +
                    self.cos(pred, pred_pad[:, :, 1:-1, 2:]) +
                    self.cos(pred, pred_pad[:, :, 2:, 1:-1]) +
                    self.cos(pred, pred_pad[:, :, 2:, :-2]) +
                    self.cos(pred, pred_pad[:, :, :-2, 2:])
            )
            return 8 - cos_sim.mean()


class MorphSimLoss(nn.Module):
    """
       cell-based Morphology Similarity loss
    """
    def __init__(self, alpha=1.2):
        super(MorphSimLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred, mask):
        num_cell = 0
        sim = torch.zeros(1, requires_grad=True).to(pred.device)[0]

        batch, _, w, h = pred.shape
        for b in range(batch):
            batch_pred = pred[b]
            index = torch.unique(mask[b])[1:]
            num_cell += len(index)
            for idx in index:
                idx_mask = mask[b, 0] == idx

                vectorx, vectory = batch_pred[0][idx_mask], batch_pred[1][idx_mask]
                sim += torch.var(vectorx, unbiased=False) + torch.var(vectory, unbiased=False)

        return sim / max(num_cell, 1)


class PhotoLoss(nn.Module):
    """
    photo loss
    """
    def __init__(self):
        super(PhotoLoss, self).__init__()

    def forward(self, x, y, occ_mask=None, photo_loss_type='abs_robust', photo_loss_delta=0.4, photo_loss_use_occ=False):
        occ_weight = occ_mask
        if photo_loss_type == 'abs_robust':
            photo_diff = x - y
            loss_diff = (torch.abs(photo_diff) + torch.tensor(0.01)).pow(photo_loss_delta)
        elif photo_loss_type == 'charbonnier':
            photo_diff = x - y
            loss_diff = ((photo_diff) ** 2 + 1e-6).pow(photo_loss_delta)
        elif photo_loss_type == 'L1':
            photo_diff = x - y
            loss_diff = torch.abs(photo_diff + 1e-6)
        # elif photo_loss_type == 'SSIM':
        #     loss_diff, occ_weight = cls.weighted_ssim(x, y, occ_mask)
        else:
            raise ValueError('wrong photo_loss type: %s' % photo_loss_type)

        if photo_loss_use_occ:
            photo_loss = torch.sum(loss_diff * occ_weight) / (torch.sum(occ_weight) + 1e-6)
        else:
            photo_loss = torch.mean(loss_diff)
        return photo_loss


