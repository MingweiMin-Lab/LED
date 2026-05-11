import numpy as np
import tifffile
import pandas as pd
import time
from multiprocessing import Process as multirun
from multiprocessing import Manager  #, Queue, JoinableQueue
from itertools import chain
import datetime
now = datetime.datetime.now()
from utils import get_cfg
import hydra
from hydra.utils import to_absolute_path as abs_path
from linear_solver import linear_solver
from os.path import join
from scipy.spatial.distance import cdist
from omegaconf import DictConfig
# from GUI import LineageTree as LT


def merging_and_pruning(cfg, track, centroid,
                        lieange_tree=False):
    """对细胞轨迹进行合并和修剪处理。

    通过合并邻近帧细胞距离较近的轨迹，以及修剪过短的轨迹，
    来优化细胞追踪结果，减少由碎片和过度分割导致的错误轨迹。

    Args:
        cfg: 配置对象，包含处理参数
        track: 轨迹数组，形状为(n_tracks, n_frames)的np.ndarray
        centroid: 细胞质心坐标数组
        lieange_tree: 是否显示家系树结构，默认为False

    Returns:
        合并和修剪后的轨迹数组，形状为(n_tracks, n_frames)的np.ndarray，
        转置后返回以使维度为(n_frames, n_tracks)
    """
    # if lieange_tree:
    #     lt = LT.LineageTree(track, if_scene=False).tree
    #     lt.show()

    # # merging the track whose neighboring-frame cells are close
    if cfg.track.merge:
        track = merge_track(cfg, track, centroid)
    # # # # prune too short tracks (caused by debris)
    mm = track[:, -1] != -1
    len_ = track.shape[1] - np.sum(track == -1, axis=1)
    ll = len_ > cfg.track.min_length
    track = track[np.logical_or(mm, ll)]

    # prune too short leaf tracks (caused by debris and over-segmentation)
    if cfg.track.prune_leaf:
        track = prune(track, cfg.track.min_length)

    return track.T


def leaf_length(track, tr):
    cell_id = np.where(tr != -1)[0]
    if cell_id.size == 0:
        return 0, 0, 0, -1
    branch = np.array([np.sum(track[:, id] == tr[id]) for id in cell_id])
    leaf = cell_id[np.where(branch == 1)[0]]
    if leaf.size == 0:
        return 0, cell_id[-1], cell_id[0], cell_id[-1]
    return leaf.max() - leaf.min() + 1, leaf.max(), leaf.min(), cell_id[-1]


def check_track(track):
    for frame, cells in enumerate(track[:-1]):
        cell_list = np.unique(cells)
        cell_list = cell_list[cell_list != -1]
        lnk_num = [len(np.unique(track[frame + 1, np.where(cells == c)[0]])) for c in cell_list]
        if np.any(np.array(lnk_num) > 2):
            return False
    return True


def prune(track, min_length=10):
    """
    裁剪短轨迹

    根据最小长度阈值删除过短的轨迹。算法会迭代查找并裁剪短轨迹，
    对于每个轨迹，找出共享相同父细胞的分支，保留较长的分支而删除较短的。

    Args:
        track: 轨迹数据，shape=(n_tracks, n_frames)，np.ndarray
        min_length: 轨迹最小长度阈值

    Returns:
        裁剪后的轨迹，shape=(n_tracks, n_frames)，np.ndarray
    """
    not_all_pruned = True
    while not_all_pruned:
        all_leaf_len = [leaf_length(track, tr) for tr in track]
        prune_trs = []
        eq_len = []
        for ith, tr in enumerate(track):

            not_max_leaf = False
            leaf_len, leaf_max, leaf_min, track_max = all_leaf_len[ith]
            branch = np.where(track[:, leaf_min-1] == tr[leaf_min-1])[0] \
                if leaf_min > 0 or tr[leaf_min-1] != -1 else np.array([])
            for i in branch:
                if i == ith:
                    continue
                leaf_len_, leaf_max_, leaf_min_, track_max_ = all_leaf_len[i]
                if track_max_ > track_max:
                    not_max_leaf = True
                    break
                elif track_max_ == track_max and leaf_len < min_length:
                    prune_trs.append(max(ith, i))
                    eq_len.append([min(ith, i), leaf_min])
            if leaf_len < min_length and not_max_leaf and leaf_max != track.shape[1] - 1:
                prune_trs.append(ith)
        track = np.delete(track, prune_trs, axis=0)
        if len(prune_trs) == 0:
            not_all_pruned = False

    return track


def merge_track(cfg, track, centroid,
                itv=1,
                merge_branch=True):
    """
    合并细胞轨迹，包括分裂轨迹的合并和跨间隔轨迹的合并

    Args:
        cfg: 配置对象，包含轨迹相关设置
        track: 轨迹数组，形状为[轨迹数量, 帧数]，存储每条轨迹在各帧中的细胞ID
        centroid: 质心信息，包含各帧中细胞的位置、大小等信息
        itv: 跨帧间隔，默认为1
        merge_branch: 是否合并分支轨迹，默认为True

    Returns:
        track: 合并后的轨迹数组，形状可能比输入小
    """
    nearest = cfg.track.nearest
    ##  itv>1 时后续分析还有bug, 请将itv设置为1
    merge_track = []
    for ith, tr in enumerate(track):
        cell_id0 = np.where(tr != -1)[0]
        frame_min, frame_max = np.min(cell_id0), np.max(cell_id0)

        for forward in range(frame_max + 1, frame_max + itv + 1):
            if forward < track.shape[1]:
                cndd_idx = np.where(track[:, frame_max] == -1)[0]
                cndd_idx1 = np.where(track[cndd_idx, forward - 1] == -1)[0]
                merge_idx = np.where(track[cndd_idx[cndd_idx1], forward] != -1)[0]
                merge_idx = cndd_idx[cndd_idx1[merge_idx]]
                if len(merge_idx) > 0:
                    # ### not consider of m
                    dist = np.linalg.norm(centroid[frame_max][tr[frame_max], 0, :2] -
                                           centroid[forward][track[merge_idx, forward], 0, :2], 2, axis=1)
                    size = centroid[frame_max][tr[frame_max], 0, 3]
                    size_list = centroid[forward][track[merge_idx, forward], 0, 3]
                    # dist1 = [np.linalg.norm(centroid[frame_max][tr[frame_max], 0, :2] -
                    #                        centroid[forward][track[idx, forward], 0, :2], 2) for idx in merge_idx]
                    # ### consider of m
                    # dist = [np.linalg.norm(centroid[frame_max][tr[frame_max], 0] -
                    #                        np.sum(centroid[forward][track[idx, forward]], axis=0), 2)
                    #                        for idx in merge_idx]

                    if_merge = np.argsort(dist)[0]
                    if nearest or (dist[if_merge] < cfg.track.thr_dist
                                   and size_list[if_merge] > 0.6 * size
                                   and size_list[if_merge] < 1.4 * size):
                        cell = track[merge_idx[if_merge], forward]
                        for idx in np.where(track[:, forward] == cell)[0]:
                            track[idx, :forward] = track[ith, :forward]
                        # print('merge', idx, 'to', ith, 'with distance', dist[if_merge], 'frame', forward)
                        merge_track.append(ith)
                        break
    track = np.delete(track, merge_track, axis=0)

    if merge_branch:
        lnk_num_track = np.zeros_like(track, dtype=np.int8) - 1
        for fth, cellss in enumerate(track.T[:-1]):
            candd_mother = np.unique(cellss)
            candd_mother = candd_mother[candd_mother != -1]
            mother_id = [np.where(track[:, fth] == m)[0] for m in candd_mother]
            for idxs in mother_id:
                if len(np.unique(track[idxs, fth + 1])) > 2:
                    print(len(idxs))
                lnk_num_track[idxs, fth] = len(np.unique(track[idxs, fth + 1]))
        assert np.all(lnk_num_track < 3), 'wrong linking !!!'
        # merge branching
        for ith, tr in enumerate(track):
            cell_id0 = np.where(tr != -1)[0]
            frame_min = np.min(cell_id0)
            if frame_min == 0:
                continue
            centx, centy, label, size = centroid[frame_min][tr[frame_min]][0]
            dist_edge = np.min((centx, centy, cfg.shape[0] - centx, cfg.shape[1] - centy))
            if dist_edge > cfg.track.thr_dist:
                dist = np.linalg.norm(centroid[frame_min - 1][:, 0, :2] - np.array((centx, centy)), 2, axis=1)
                size_list = centroid[frame_min - 1][:, 0, 3]
                # size_diff = size - size_list
                # size_diff_rel = np.abs(size_diff) / size
                candd = np.argsort(dist)
                ##  debug: 排除已有两个连接的细胞
                for id in candd:
                    # if id in candd_mother_id:
                    trackid = np.where(track[:, frame_min - 1] == id)[0]
                    cell_cycle = np.all(lnk_num_track[trackid,
                                        frame_min - cfg.track.div_interval: frame_min + cfg.track.div_interval] == 1)
                    if nearest or (cell_cycle
                                   and len(trackid) > 0
                                   and dist[id] < cfg.track.thr_dist
                                   and size_list[id] < 0.7*size
                                   and size_list[id] > 0.3*size):
                        # trackid = np.where(track[:, frame_min - 1] == candd)[0]
                        mergeid = track[:, frame_min] == track[ith, frame_min]
                        track[mergeid, :frame_min] = track[trackid[0], :frame_min]
                        lnk_num_track[trackid, frame_min - 1] += 1
                        lnk_num_track[mergeid, frame_min - 1] = lnk_num_track[trackid[0], frame_min - 1]
                        break
        assert np.all(lnk_num_track < 3), 'wrong linking !!!'

    assert check_track(track.T)
    return track


def get_interference(num_cell, centroid, center,
                     radius=200, a=0.4, b=1, c=0):
    """
    计算细胞间的干扰概率

    基于细胞总数和邻居数量计算细胞从一个位置移动到另一个位置的干扰概率。
    公式: P(x_(t+1,i)↛x_(t,j)) = 1/(α×total cell count + neighbor count + c)

    Args:
        num_cell: 细胞总数
        centroid: 细胞的质心坐标数组，形状为(n, 2)，包含每个细胞的x, y坐标
        center: 中心点坐标，用于计算距离的参考中心
        radius: 邻居搜索半径，默认为200，仅在此范围内的细胞计入邻居数
        a: 总细胞数权重系数，默认为0.4
        b: 邻居数权重系数，默认为1
        c: 偏置常数，默认为0

    Returns:
        float: 干扰概率值，值越小表示干扰越强
    """
    points_array = np.array(centroid[:, :, :2])
    center_array = np.array(center)
    # the distance from each point to the center
    distances = np.linalg.norm(points_array - center_array, axis=1)
    # Count the number of cells within the radius
    num_neighbor = np.sum(distances <= radius)

    # P(x_(t+1)↛x_(t) )=1/(α×total cell count+neighbor count+1)
    intf = 1 / (a * num_cell + b * num_neighbor + c)
    return intf


def expand_virtual_node(theta, matrix, centroid, shape):
    """
    为新进入视野的细胞添加虚拟节点到转移矩阵。

    Args:
        theta: 高斯函数参数，控制距离衰减速度。
        matrix: 转移矩阵。
        centroid: 细胞的质心坐标，形状为 (N, 2, 4)。
        shape: 图像的形状 (height, width)。

    Returns:
        扩展并转置后的转移矩阵，最后一列为虚拟节点的值。
    """
    # trans_matrix = np.hstack((matrix, np.zeros((matrix.shape[0], 1), dtype=np.float32)))

    # for i, cent in enumerate(centroid[:, 0, :2]):
    #
    #     dist_edge = np.min((cent[0], cent[1], shape[0]-cent[0], shape[1]-cent[1]))
    #     dist_edge_gau = np.exp(-dist_edge ** 2 / theta)
    #
    #     dist_neighbor = np.max(matrix[i])
    #     # # # if dist_edge < 100 else -1 这会增加错误链接
    #     trans_matrix[i][-1] = np.max((dist_edge_gau, 1 - dist_neighbor))  # if dist_edge < 100 else -1

    cent = centroid[:, 0, :2]
    dist_edge = np.min([cent[:, 0], cent[:, 1], shape[0]-cent[:, 0], shape[1]-cent[:, 1]], axis=0)
    dist_edge_gau = np.exp(-dist_edge ** 2 / theta)
    dist_neighbor = np.max(matrix, axis=1)
    virtual_node = np.max((dist_edge_gau, 1 - dist_neighbor), axis=0)
    trans_matrix = np.hstack((matrix, virtual_node.reshape(-1, 1)))

    return trans_matrix.T


def cal_trans_matrix(cfg,
                     centroid: list,
                     mask,
                     flow,
                     num_cell_f,
                     theta: float=800.,
                     jitter_thr: float=0.6,
                     global_norm: bool=True,
                     ):
    """
    计算两帧之间的转移矩阵，用于细胞跟踪。

    根据光流信息和质心位置，计算从前一帧到当前帧的细胞转移概率矩阵。
    支持全局归一化和局部归一化两种计算模式。

    Args:
        cfg: 配置对象，包含跟踪参数（如max_movenment）
        centroid: 质心列表，centroid[-1]为当前帧，centroid[-2]为前一帧
        mask: 当前帧的分割掩码
        flow: 光流数据，包含速度分量
        num_cell_f: 每帧的细胞数量列表
        theta: 距离衰减参数，控制转移概率的衰减速度，默认800
        jitter_thr: 抖动阈值，低于此比例的位移视为抖动并置零，默认0.6
        global_norm: 是否使用全局归一化模式，默认True

    Returns:
        转移概率矩阵，表示从当前帧细胞到前一帧细胞的转移概率
    """
    num_pix = np.sum(mask > 0)
    vel = np.sum(np.sum(flow[0:2], axis=1), axis=1).astype(np.float32) / num_pix
    mag = np.sum(np.sqrt((flow[0] ** 2 + flow[1] ** 2))) / num_pix

    if global_norm:
        if np.sqrt(np.sum(vel ** 2)) / mag < jitter_thr:
            vel = 0
        vj, mo = centroid[-1][:, 0, :2], centroid[-1][:, 1, :2]
        vk = centroid[-2][:, 0, :2]
        dist_lh = cdist(vj + vel, vk, metric='sqeuclidean')
        prior_lh = cdist(vj + mo, vk, metric='sqeuclidean')
        post = np.exp(-dist_lh / theta) * np.exp(-prior_lh / theta * 4)
        sum_norm = 1 / (np.sum(post, axis=1, keepdims=True) + 1e-8)
        matrix = post * sum_norm
    else:
        matrix = []
        for j, (vj, m) in enumerate(centroid[-1]):
            vj, m = vj[:2], m[:2]
            # 计算干扰 越小越均匀， 越大差异越大
            intf = get_interference(num_cell_f[-1], centroid[-1], vj, radius=cfg.shape[0] / 6)
            for k, (vk, _) in enumerate(centroid[-2]):
                vk = vk[:2]
                if np.sqrt(np.sum(vel ** 2)) / mag < jitter_thr:
                    vel = 0

                dt = np.sum((vj - vk) ** 2)
                ## 距离太大的直接不考虑, 给极小的负分数
                if dt > cfg.track.max_movenment ** 2:
                    matrix.append(-1)
                else:
                    dist = np.sum((vj + vel - vk) ** 2)
                    dist_lh = np.exp(-dist / theta)  # / (theta * np.sqrt(2 * np.pi))  ##  快速计算，不算常数因子，不影响结果
                    dist = np.sum((vj + m - vk) ** 2)
                    prior = np.exp(-dist / theta * 2)  # / (theta * np.sqrt(2 * np.pi))    ##  快速计算，不算常数因子，不影响结果
                    post = prior * dist_lh / (prior * dist_lh + (1 - prior) * intf)
                    matrix.append(post)
        matrix = np.array(matrix, dtype=np.float32).reshape(len(centroid[-1]), len(centroid[-2]))
    matrix = expand_virtual_node(theta, matrix, centroid[-1], cfg.shape)

    return matrix


def get_track(cfg, q,
              mask_dir: list,
              flow_dir: list,
              run_id: int,
              max_run: int,
              jitter_thr=0.6,
              max_division:int=None,
              new_detect:int=None,
              ) -> None:
    """
    获取跟踪过渡矩阵

    读取掩码和光流数据，计算细胞质心，过渡矩阵，并通过线性求解器进行细胞匹配。

    Args:
        cfg: 配置对象
        q: 多进程队列
        mask_dir: 掩码文件路径列表
        flow_dir: 光流文件路径列表
        run_id: 运行 ID
        max_run: 最大运行次数
        jitter_thr: 抖动阈值，默认为 0.6
        max_division: 最大分裂数，可选
        new_detect: 新检测参数，可选

    Returns:
        None，结果通过队列返回
    """
    print(f'------{run_id}th running start-------\n')
    num_cell_f = []
    centroid = []
    trans_matrix = []
    match = []

    assert len(mask_dir) >= 2,  'mask dir should be longer than 2'
    assert len(flow_dir) >= 1,  'flow dir should be longer than 1'

    mask = tifffile.imread(mask_dir[0])
    labels = np.unique(mask)[1:]

    centroidt0 = []
    for label in labels:
        cell_p = np.where(mask == label)
        pix_ = len(cell_p[0])
        centroidt0.append((np.append(np.mean(np.asarray(cell_p).T, axis=0),
                                     (label, pix_)), (-1, -1, -1, -1)))
    assert len(labels) > 0, f'no cell detected in 0th frame!!! cell num = {len(labels)}'
    centroid.append(np.asarray(centroidt0))
    num_cell_f.append(len(centroid[-1]))
    theta = cfg.track.max_movenment ** 2 / 16

    for i in range(len(flow_dir)):
        mask = tifffile.imread(mask_dir[i+1])
        labels = np.unique(mask)[1:]
        centroidt = []
        assert len(labels) > 0, f'no cell detected in {i}th frame!!! cell num = {len(labels)}'
        # assert len(labels) == labels.max(), f"number and id of cells doesn't match"
        flow = tifffile.imread(flow_dir[i])
        flow = flow * (mask > 0)
        for label in labels:
            pix_ = len(np.where(mask == label)[0])
            cell = np.where(mask == label)
            m = np.append(np.mean(flow[:2, cell[0], cell[1]], axis=1), (0, 0))
            centroidt.append((np.append(np.mean(np.asarray(cell).T, axis=0), (label, pix_)), m))
        centroid.append(np.asarray(centroidt))
        num_cell_f.append(len(centroid[-1]))
        matrix = cal_trans_matrix(cfg, centroid, mask, flow, num_cell_f, theta, jitter_thr)
        final_match = linear_solver(cfg, matrix, max_division=max_division, new_detect=new_detect)
        match.append(final_match)
    if run_id < max_run - 1:
        q.put([run_id, num_cell_f[:-1], centroid[:-1], trans_matrix, match])
    else:
        q.put([run_id, num_cell_f, centroid, trans_matrix, match])
    print(f'queue size: {q.qsize()}/{max_run}', f'------{run_id}th running end-------\n')


def run(cfg, q, mask_dir, flow_dir,
        run_num, start_frame, num_f,
        last_itv=10, jitter_thr=0.6, type='notequal'):
    """
    多线程/多进程运行细胞追踪任务。

    根据 run_num 参数将任务分配到多个线程并行处理，支持两种分配方式：
    - equal 分配：每个线程处理相近数量的文件
    - 加权分配：按递增间隔分配，后续线程处理更多文件

    Args:
        cfg: 配置文件对象
        q: 队列或数据队列
        mask_dir: 掩码文件列表
        flow_dir: 光流文件列表
        run_num: 并行线程数量
        start_frame: 起始帧编号
        num_f: 总帧数
        last_itv: 加权分配时第一个线程处理的帧数，默认10
        jitter_thr: 抖动阈值，默认0.6
        type: 分配类型，'equal' 为均分分配，其他值为加权分配，默认 'not equal'

    Returns:
        None
    """
    rs = []
    if run_num <= 1:
        get_track(cfg, q, mask_dir, flow_dir, 0, 1, jitter_thr)    # 单线程处理
    elif type == 'equal' or run_num < 10:# or run_num < 4:
        # 计算每个线程需要处理的文件数量
        interval = num_f // run_num + 1
        if run_num > num_f:
            run_num = num_f - 1
            interval = 1
        for i in range(run_num):
            if int(i * interval) < len(mask_dir)-1:
                args = (cfg, q,
                        mask_dir[int(i * interval):int(i * interval + interval + 1)],
                        flow_dir[int(i * interval):int(i * interval + interval)],
                        i,
                        run_num,
                        jitter_thr
                        )
                r = multirun(target=get_track, args=args)
                r.start()
                rs.append(r)
    else:
        if run_num > num_f:
            run_num = num_f - 1
            itvs = np.array([1] * run_num).astype(int)
        else:
            itv = (2 * num_f / run_num - 2 * last_itv) / (run_num - 1)
            itvs = [int(np.round(last_itv + i * itv)) for i in range(run_num)][::-1]

            itvs[0] = int(itvs[0] + num_f - np.sum(itvs) - 1)
            itvs = np.asarray(itvs)
        print('file intervals:', itvs, f'sum: {np.sum(itvs)} + 1')
        assert np.sum(itvs) == num_f-1, f'intervals sum {np.sum(itvs)} is not equal to cell num {num_f} - 1'
        assert np.all(np.array(itvs) > 0), f'there is zero-interval '

        for i in range(run_num):
            print(f'file interval {i}:',
                  [start_frame+np.sum(itvs[:i]), start_frame+np.sum(itvs[:i+1])],
                  len(mask_dir[np.sum(itvs[:i]): np.sum(itvs[:i+1])]))
            args = (cfg, q,
                    mask_dir[np.sum(itvs[:i]): np.sum(itvs[:i+1]) + 1],
                    flow_dir[np.sum(itvs[:i]): np.sum(itvs[:i+1])],
                    i,
                    run_num,
                    jitter_thr,
                    )
            r = multirun(target=get_track, args=args)
            r.start()
            rs.append(r)
    [r.join() for r in rs]
    print(f'------all running ended-------\n')


def get_lineage(cfg, tracks, num_cell_f):
    """
    根据细胞轨迹数据构建细胞谱系。

    遍历所有帧，从初始帧开始逐帧构建每个细胞的谱系链。
    支持细胞的持续存在、消失（标记为-1）和新出现的轨迹。

    Args:
        cfg: 配置对象
        tracks: 轨迹列表，每帧包含该帧的细胞ID列表
        num_cell_f: 每帧的细胞数量列表

    Returns:
        numpy.ndarray: 转置后的细胞谱系数组
    """

    for f, tr in enumerate(tracks):

        # 提取当前轨迹中的细胞id
        cells_t = set(tr)

        assert len(tr) == num_cell_f[f + 1], \
            f"number {len(tr)} and id {num_cell_f[f + 1]} don't match"

        current_tracks = []
        # 初始化当前迭代的轨迹列表
        if f == 0:
            all_tracks = [[i] for i in range(num_cell_f[0])]

        # 遍历当前已有的轨迹
        for track in all_tracks:

            # 提取最后一个id
            last_id = track[-1]

            next_cells = np.where(np.array(tr) == last_id)[0]

            # 如果id不在当前细胞集合中，则标记为-1（死亡、出视野）
            if last_id not in cells_t:
                current_tracks.append(track + [-1])
                assert next_cells.size == 0
                # continue
            else:
                # 为每个可能的下一个轨迹点连接到当前轨迹上
                for next_cell in next_cells:
                    current_tracks.append(track + [next_cell])

        for next_cell in np.where(np.array(tr) == num_cell_f[f])[0]:
            current_tracks.append([-1] * (f + 1) + [next_cell])

        # 注意：这里将current_tracks赋值回all_tracks，累积所有轨迹
        all_tracks = current_tracks
    track_list = np.array(all_tracks, dtype=np.int32)[:-1].T

    assert check_track(track_list)

    return np.array(all_tracks, dtype=np.int32).T


def cell_tracker(cfg,
                 mask_file,
                 flow_file,
                 run_num=1,
                 start_frame=0,
                 num_f=100,
                 last_itv=2,
                 jitter_thr=0.6,
                 ):
    """
    细胞跟踪主程序入口。

    使用多线程方式处理细胞掩膜和光流数据，执行细胞跟踪并返回轨迹和质心结果。

    Args:
        cfg: 配置对象，包含跟踪算法的相关配置参数。
        mask_file: 细胞掩膜文件路径。
        flow_file: 光流文件路径。
        run_num: 运行次数，默认为1。
        start_frame: 起始帧索引，默认为0。
        num_f: 处理的总帧数，默认为100。
        last_itv: 最后时间间隔，默认为2。
        jitter_thr: 抖动阈值，默认为0.6。

    Returns:
        tuple: 包含轨迹数组和质心数组的元组。
    """
    run_id, num_cell_f, centroid, trans_matrix, track = [], [], [], [], []
    with Manager() as manager:
        q = manager.Queue()
        run(cfg, q, mask_file, flow_file, run_num, start_frame, num_f, last_itv, jitter_thr, type='not equal')
        while not q.empty():
            n, nc, cn, tm, tr = q.get()
            run_id.append(n)
            num_cell_f.append(nc)
            centroid.append(cn)
            # trans_matrix.append(tm)
            track.append(tr)
    num_cell_f = [num_cell_f[i] for i in sorted(range(len(run_id)), key=lambda k: run_id[k])]
    num_cell_f = list(chain(*num_cell_f))
    centroid = [centroid[i] for i in sorted(range(len(run_id)), key=lambda k: run_id[k])]
    centroid = list(chain(*centroid))
    # trans_matrix = [trans_matrix[i] for i in sorted(range(len(run_id)), key=lambda k: run_id[k])]
    # trans_matrix = list(chain(*trans_matrix))
    track_match = [track[i] for i in sorted(range(len(run_id)), key=lambda k: run_id[k])]
    track_match = list(chain(*track_match))

    # if method == 'greedy':
    #     import greedy
    #     track_cell = greedy.track_cell
    # elif method == 'linear_solver':
    #     import linear_solver
    #     track_cell = linear_solver.track_cell
    #
    # else:
    #     raise ValueError("unsupported method")
    np.save(join(cfg.result, now.strftime('%Y-%m-%d-') + 'centroid.npy'), np.array(centroid, dtype=object))
    track = get_lineage(cfg, track_match, num_cell_f)

    return track, centroid


def convert2id(cfg, track, centroid):
    track_copy = np.zeros_like(track, dtype=np.int16) - 1
    for ith, tr in enumerate(track):
        for jth, t in enumerate(tr):
            if track[ith, jth] != -1:
                track_copy[ith, jth] = int(centroid[jth][track[ith, jth], 0, 2])
    df = pd.DataFrame(data=track_copy.T, columns=None)
    df.to_csv(join(cfg.result, now.strftime("%Y-%m-%d") + 'track_id.csv'), index=False)


@hydra.main(config_path=abs_path('config'), version_base='1.3', config_name='tracker')
def tracker(cfg: DictConfig):
    """
        主程序入口，用于启动多线程处理过程。
        Args:
        method: 选择的跟踪方法，可以是'greedy', 'linear_solver'之一
        mask_dir: 细胞掩膜文件夹路径，
        """
    cfg = get_cfg(cfg, type='track')
    start = time.time()

    run_num = cfg.track.run_num
    last_itv = cfg.track.last_itv

    start_frame = cfg.dataloader.start_frame
    num_f = cfg.dataloader.num_frame

    if cfg.track.load_track:
        track = pd.read_csv(abs_path(cfg.track.track_file)).to_numpy()
        centroid = np.load(abs_path(cfg.track.centroid_file), allow_pickle=True)
    else:
        track, centroid = cell_tracker(cfg,
                                       mask_file=cfg.mask_dir,
                                       flow_file=cfg.flow_dir,
                                       run_num=run_num,
                                       start_frame=start_frame,
                                       num_f=num_f,
                                       last_itv=last_itv,
                                       jitter_thr=cfg.track.jitter_thr,
                                       )
    if num_f > 15 and cfg.track.post_pro:
        track = merging_and_pruning(cfg, track.T, centroid)
    df = pd.DataFrame(data=track, columns=None)
    df.to_csv(join(cfg.result, now.strftime("%Y-%m-%d") +
                   'track_results.csv'), index=False)
    convert2id(cfg, track.T, centroid)

    print(now.strftime("%Y-%m-%d %H:%M:%S") + f'{num_f}-frame time cost:', int(time.time() - start), 's')


if __name__ == "__main__":

    # tracker()
    track = pd.read_csv(abs_path(r"D:\Data\Haishan\experiment\ipsc\result00013\2025-08-25track_linear_solver_merged_and_pruned.csv")).to_numpy()
    centroid = np.load(abs_path(r"D:\Data\Haishan\experiment\ipsc\result00013\2025-08-25-centroid.npy"), allow_pickle=True)
    track = merging_and_pruning(track.T, centroid, min_length=10, prune_leaf=True, thr_merge=30)
    df = pd.DataFrame(data=track, columns=None)
    df.to_csv(join(r'D:\Data\Haishan\track\result', now.strftime("%Y-%m-%d") +
                   'track_linear_solver_merged_and_pruned.csv'), index=False)
