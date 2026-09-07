import cv2
import numpy as np
import sys
from skimage.feature import peak_local_max
from scipy.optimize import linear_sum_assignment


class Cell:
    def __init__(self, x, y, sigmaX, sigmaY, theta, intensity, mask, size, intersection):
        self.mask = mask
        self.coor = np.array([x, y])
        self.sigmaX, self.sigmaY, self.theta, self.intensity = sigmaX, sigmaY, theta, intensity
        self.size, self.intersection = size, intersection



class DivisionDetector:
    def __init__(self, max_dt, image, mask=None, mask0=None, local_maxima=None, para=None, type='seg'):
        self.image, self.local_maxima, self.para = image, local_maxima, para
        self.mask = mask
        self.mask0 = mask0
        self.x, self.y = np.indices(image.shape)
        self.sigma = max_dt / 2
        self.imgsize = image.size

        self.type = type
        self.cell_list = []
        self.get_cell_list()
        self.minsize = max(min([cell.size for cell in self.cell_list]), 50) if len(self.cell_list) > 0 else 50


    def get_cell_list(self):
        if self.type == 'seg':
            idxs = np.unique(self.mask)[1:]
            for idx in idxs:
                mask = self.mask == idx
                x_c, y_c = np.mean(np.where(mask), axis=1)
                intensity = np.mean(self.image[mask])
                intersection = np.sum(np.logical_and(mask, self.mask)) / np.sum(mask)
                if intensity > 0.1:
                    cell = Cell(x_c, y_c, 0, 0, 1, intensity, mask, np.sum(mask), intersection)
                    self.cell_list.append(cell)

        elif self.type == 'blob':
            for x_c, y_c, ith in self.local_maxima:
                sigmaX, sigmaY, theta = self.para[ith]
                if sigmaY - sigmaX < 6 and sigmaX + sigmaY < 15:
                    mask = (((self.x - x_c) * np.cos(theta) - (self.y - y_c) * np.sin(theta)) / sigmaX) ** 2 + \
                           (((self.x - x_c) * np.sin(theta) + (self.y - y_c) * np.cos(theta)) / sigmaY) ** 2 < 1
                    intensity = (mask * self.image).sum() / mask.sum()

                    if intensity > 0.1:
                        cell = Cell(x_c, y_c, sigmaX, sigmaY, theta, intensity, mask, np.pi*sigmaX*sigmaY, 1)
                        self.cell_list.append(cell)

    def prefer_score(self, cell1, cell2):
        """
        计算两个细胞之间的分裂分数

        基于多个因素综合评估两个细胞是否适合合并，包括距离、角度、强度、大小和交集面积。

        Args:
            cell1: 第一个细胞对象，包含坐标、大小、强度等属性
            cell2: 第二个细胞对象，包含坐标、大小、强度等属性

        Returns:
            float: 优先分数，值越高表示两个细胞越匹配
        """
        dt_mean = self.sigma
        dt_std = 10
        # w = (dt_std * np.sqrt(2 * np.pi))
        dt = np.sum((cell1.coor - cell2.coor) ** 2) ** 0.5
        if dt > dt_mean * 3 or dt < 5:
            prefer_score = 0
        else:
            dt_score = np.exp(-0.5 * ((dt - dt_mean) / dt_std) ** 2)  # / w
            if cell1.sigmaX == cell1.sigmaY or cell2.sigmaY == cell2.sigmaX:
                theta_score = 1
            else:
                theta_score = np.cos(cell1.theta - cell2.theta)
            # sigma_s = np.exp(-np.abs(cell1.sigmaX - cell2.sigmaX) - np.abs(cell1.sigmaY - cell2.sigmaY) - \
            #         np.abs(cell1.sigmaY - cell2.sigmaX) - np.abs(cell1.sigmaX - cell2.sigmaY))
            # sigma_score = sigma_s * (np.exp(-cell1.sigmaX) + np.exp(-cell1.sigmaY) +
            #                          np.exp(-cell2.sigmaX) + np.exp(-cell2.sigmaY))
            intensity_score = (cell1.intensity + cell2.intensity) * np.abs(cell1.intensity - cell2.intensity)
            if np.abs(cell1.size - cell2.size) > self.minsize * 2 or cell1.size > self.minsize * 3 or cell2.size > self.minsize * 3:
                size_score = 0
            else:
                size_score = np.exp(-(cell1.size - cell2.size)**2 / self.imgsize) * \
                             np.exp(-(cell1.size - self.minsize)**2 / self.imgsize) * \
                             np.exp(-(cell2.size - self.minsize)**2 / self.imgsize)

            intersection_score = min(cell1.intersection + cell2.intersection, 1)

            prefer_score = dt_score * theta_score * intensity_score * size_score * intersection_score  # * sigma_score
        return prefer_score

    def division_match(self):
        """
        使用匈牙利算法检测细胞分裂匹配

        基于细胞的特征分数构建成本矩阵，通过匈牙利算法找到最优配对，
        筛选出有效的细胞分裂对，并生成分割掩码。

        Returns:
            division_mask: 细胞分裂的掩码
        """

        n = len(self.cell_list)
        self.cost_matrix = np.zeros((n, n))

        # 构建成本矩阵
        for i in range(n):
            for j in range(n):
                if i == j:
                    self.cost_matrix[i, j] = 0  # 避免自匹配
                else:
                    self.cost_matrix[i, j] = self.prefer_score(self.cell_list[i], self.cell_list[j])

        # 使用匈牙利算法找到最小成本匹配
        row_ind, col_ind = linear_sum_assignment(-self.cost_matrix)

        # 筛选有效配对
        pairs = []
        link_idx = []
        for i, j in zip(row_ind, col_ind):
            if self.cost_matrix[i, j] > 0 and i > j:
                if i not in link_idx and j not in link_idx:
                    link_idx.append(i)
                    link_idx.append(j)
                    pairs.append((i, j))
                elif i in link_idx:
                    where = np.where(np.array(pairs) == i)[0][0]
                    old_pairs = pairs[where]
                    if self.cost_matrix[i, j] > self.cost_matrix[old_pairs]:
                        link_idx.append(i)
                        link_idx.append(j)
                        pairs[where] = (i, j)
                elif j in link_idx:
                    where = np.where(np.array(pairs) == j)[0][0]
                    old_pairs = pairs[where]
                    if self.cost_matrix[i, j] > self.cost_matrix[old_pairs]:
                        link_idx.append(i)
                        link_idx.append(j)
                        pairs[where] = (i, j)

        division_mask = self.get_division_mask(pairs)

        return division_mask

    def get_division_mask(self, pairs):
        division_mask = np.zeros_like(self.image, dtype=np.float32)
        for cells in pairs:
            cell1, cell2 = self.cell_list[cells[0]], self.cell_list[cells[1]]
            prefer = self.cost_matrix[cells[0], cells[1]]

            division_mask[cell1.mask] = -prefer
            division_mask[cell2.mask] = prefer

        return division_mask


class Ellip_log_blob():
    
    def get_ellip_log_kernel(self, kernelSize, sigma_x, sigma_y, theta):
        """
        生成旋转的Laplacian of Gaussian (LoG) 滤波器核。

        Args:
            kernelSize: 核尺寸，可以是标量或向量 [行, 列]
            sigma_x: x方向的标准差
            sigma_y: y方向的标准差
            theta: 旋转角度（弧度）

        Returns:
            ndarray: 计算得到的LoG滤波器核
        """

        siz = (kernelSize - 1) // 2

        a = np.cos(theta) ** 2 / 2 / sigma_x ** 2 + np.sin(theta) ** 2 / 2 / sigma_y ** 2
        b = -np.sin(2 * theta) / 4 / sigma_x ** 2 + np.sin(2 * theta) / 4 / sigma_y ** 2
        c = np.sin(theta) ** 2 / 2 / sigma_x ** 2 + np.cos(theta) ** 2 / 2 / sigma_y ** 2

        x = y = np.linspace(-siz, siz, 2*siz+1)
        x, y = np.meshgrid(x, y)
        arg = - (a * x**2 + 2 * b * x * y + c * y**2)

        h = np.exp(arg)
        h[h < sys.float_info.epsilon * h.max()] = 0

        h = h/h.sum() if h.sum() != 0 else h

        # calculate Laplacian
        lapl = ((2 * a * x + 2 * b * y)**2 - 2 * a) + ((2 * c * y + 2 * b * x)**2 - 2 * c)
        h1 = h * lapl
        return h1

    def LoG_filter(self, largestSigma, smallestSigma, sigmaStep, thetaStep):
        """
        生成多尺度椭圆高斯拉普拉斯(LoG)滤波器核。

        通过在不同的sigmaX、sigmaY和theta参数下生成LoG核，构建一个多尺度的
        椭圆blob检测滤波器组，用于检测不同尺度、不同方向的椭圆结构。

        Args:
            largestSigma: 最大sigma值，定义检测的最大尺度。
            smallestSigma: 最小sigma值，定义检测的最小尺度。
            sigmaStep: sigma步进数，控制sigmaX和sigmaY的采样密度。
            thetaStep: theta步进数，控制旋转角度的采样密度。

        Returns:
            filter_kernel: 生成的LoG滤波器核数组，形状为(n, kernelSize+1, kernelSize+1)。
            para_list: 对应的参数列表，每行为(sigmaX, sigmaY, theta)。
            kernel_mask: 二值核掩码，用于标识有效核区域。
        """

        kernelSize = largestSigma * 4
        para_list = np.zeros((int((sigmaStep-1)*sigmaStep*thetaStep/2+sigmaStep), 3))
        filter_kernel = np.zeros((int((sigmaStep-1)*sigmaStep*thetaStep/2+sigmaStep), kernelSize + 1, kernelSize + 1))  # circular multi-scale gLoG kernel
        kernel_mask = np.zeros_like(filter_kernel)
        x, y = np.indices(kernel_mask.shape[1:])
        x_c = kernel_mask.shape[-2] // 2
        y_c = kernel_mask.shape[-1] // 2
        k = 0
        for i, sigmaX in enumerate(np.linspace(smallestSigma, largestSigma, sigmaStep)):
            for sigmaY in np.linspace(sigmaX, largestSigma, sigmaStep-i):
                for ith, theta in enumerate(np.linspace(0.0, np.pi, thetaStep, endpoint=False)):
                    if sigmaX == sigmaY and ith > 0:
                        break
                    # print('sigmaX, sigmaY, theta:', sigmaX, sigmaY, theta)
                    filter_kernel[k] = -self.get_ellip_log_kernel(filter_kernel.shape[1], sigmaX, sigmaY, theta)
                    # filter_kernel[k] = filter_kernel[k] * (1 + np.log(sigmaX**alpha)) * (1 + np.log(sigmaY**alpha))
                    # filter_kernel[k] = filter_kernel[k] * sigmaX * sigmaY
                    theta = np.pi / 2 - theta
                    para_list[k] = np.array((sigmaX, sigmaY, theta))
                    kernel_mask[k] = (((x - x_c)*np.cos(theta)-(y - y_c)*np.sin(theta))/sigmaX) ** 2 + \
                                     (((x - x_c)*np.sin(theta)+(y - y_c)*np.cos(theta))/sigmaY) ** 2 < 4
                    mean = (filter_kernel[k]*kernel_mask[k]).sum()/kernel_mask[k].sum()
                    filter_kernel[k] = filter_kernel[k] - mean
                    k = k + 1
        # tifffile.imwrite('filter_kernel.tif', filter_kernel)
        return filter_kernel, para_list, kernel_mask

    def ellip_blob(self, image, largestSigma=9,  smallestSigma=5, sigmaStep=5, thetaStep=8, exclude_border=False):
        """使用拉普拉斯高斯 (LoG) 方法在灰度图像中检测斑点。

        该方法通过LoG滤波器检测图像中的斑点，并返回每个斑点的坐标以及检测到该斑点的高斯核标准差。

        Args:
            image: 输入的灰度图像，假设背景为暗色（黑底白点）。
            largestSigma: 最大高斯sigma值，默认为9。
            smallestSigma: 最小高斯sigma值，默认为5。
            sigmaStep: sigma步长，默认为5。
            thetaStep: 角度步长，默认为8。
            exclude_border: 是否排除边界，默认为False。

        Returns:
            image: 原图。
            local_maxima: 检测到的局部最大值坐标。
            para: 对应的参数列表。
        """

        kernel_list, para, kernel_mask = self.LoG_filter(largestSigma=largestSigma, smallestSigma=smallestSigma,
                                                     sigmaStep=sigmaStep, thetaStep=thetaStep)
        # computing gaussian laplace
        image_cube = np.empty(image.shape + (len(kernel_list),), dtype=np.float32)
        for i, c in enumerate(kernel_list):
            image_cube[..., i] = cv2.filter2D(image, ddepth=cv2.CV_32F, kernel=kernel_mask[i]*c) * para[i, 0] * para[i, 1]

        local_maxima = peak_local_max(
            image_cube,
            min_distance=20,
            threshold_abs=image_cube.min() + image_cube.max() / 2 + 0.01,
            threshold_rel=image_cube.max() / 2 + 0.01,
            exclude_border=exclude_border,
            footprint=np.ones((5, 5, image_cube.shape[-1])),
        )

        # Catch no peaks
        if local_maxima.size == 0:
            return np.zeros_like(image), [], []

        return image, local_maxima, para

        # ## max_dt, image, mask=None, local_maxima=None, para=None, type='seg'
        # Div_det = DivisionDetector(self.max_dt, image, local_maxima, para, type='blob')
        # division_mask = Div_det.division_match()
        #
        # return division_mask

def main(imgs_dir,
         masks_dir,
         save_path,
         itv=1,
         max_dt=30,
         dettype='seg',
         # max_int=2048,
         # min_int=0,
         ):
    import tifffile
    from tifffile import imread
    from skimage import morphology
    from skimage.morphology import square, opening

    # def max_min_morn(img, max, min):
    #     return (img - min) / (max - min)
    det = Ellip_log_blob()
    for idx in range(len(imgs_dir) - 1):
        img2_file = imgs_dir[idx + itv]
        img2 = imread(img2_file)  # [0]
        if dettype == 'blob':
            img1_file = imgs_dir[idx]
            img1 = imread(img1_file)  # [0]

            # if img1.max() > 1:
            #     img1 = max_min_morn(img1, max_int, min_int)
            #     img2 = max_min_morn(img2, max_int, min_int)

            im_diff = img2 - img1
            im_nxt = im_diff > 0.1

            im_nxt = opening(im_nxt, square(3))
            im_nxt = morphology.remove_small_holes(im_nxt, area_threshold=100, connectivity=1)
            im_nxt = morphology.remove_small_objects(im_nxt, min_size=80, connectivity=1)

            # plt.imshow(im_nxt)
            # plt.show()

            assert img1.shape == img2.shape, \
                f'Images are not the same size, {img1.size} and {img2.size}'

            image, local_maxima, para = det.ellip_blob(im_nxt * img2)

            # ## max_dt, image, mask=None, local_maxima=None, para=None, type='seg'
            Div_det = DivisionDetector(max_dt, image, local_maxima, para, type=dettype)
            division_mask = Div_det.division_match()

        if dettype == 'seg':
            mask0 = imread(masks_dir[idx])
            mask = imread(masks_dir[idx + itv])  # [0]
            Div_det = DivisionDetector(max_dt, img2, mask, mask0, type=dettype)
            division_mask = Div_det.division_match()

        # print(idx, 'frame done!', division_mask.max())
        tifffile.imwrite(os.path.join(save_path, 'division' + str(idx).zfill(4) + '.tif'), division_mask)


if __name__ == '__main__':
    from hydra.utils import to_absolute_path as abs_path
    from os.path import join
    from glob import glob
    from pathlib import Path
    import tifffile

    mask_path = 'data\segment\stardist'
    imgs_path = 'data\img'
    masks_dir = np.sort(glob(abs_path(join(Path.cwd().parent, mask_path, '*.tif')))).tolist()[280:]

    imgs_dir = np.sort(glob(abs_path(join(Path.cwd().parent, imgs_path, '*.tif')))).tolist()[280:]

    main(imgs_dir, masks_dir, 'dm', dettype='seg')



