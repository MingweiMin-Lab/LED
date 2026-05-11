import os
import numpy as np
import pandas as pd
import tifffile
# from traccuracy import run_metrics
# from traccuracy.loaders import load_ctc_data
# from traccuracy.matchers import CTCMatcher, IOUMatcher
# from traccuracy.metrics import CTCMetrics, DivisionMetrics

import LineageTree as LT


def lineage_mask(mask_dir, tracks, centroid, save_path, res_path, saveRGB=False):
    """
    将细胞追踪结果转换为CTC格式并生成追踪掩码图像。

    该函数读取细胞掩码图像，根据追踪信息生成谱系树，并将结果保存为CTC格式的
    追踪文件和掩码图像。支持生成RGB彩色可视化图像或标准CTC格式图像。

    Args:
        mask_dir: 输入掩码图像目录路径
        tracks: 细胞追踪数据，形状为 (N, T) 的数组
        centroid: 细胞质心坐标信息
        save_path: 输出结果保存的根目录路径
        res_path: 结果子目录名称
        saveRGB: 是否保存RGB彩色可视化图像，默认为False

    Returns:
        无返回值，结果直接保存到指定目录
    """

    ## 检查路径是否存在，不存在则创建
    if not os.path.exists(os.path.join(save_path, res_path)):
        os.makedirs(os.path.join(save_path, res_path))
    if not os.path.exists(os.path.join(save_path, "rgb"+res_path)):
        os.makedirs(os.path.join(save_path, 'rgb'+res_path))

    # generate lineage in CTC format
    lineage = []
    track = []
    lt = LT.LineageTree(tracks, np.linspace(0, tracks.shape[-1]-1, tracks.shape[-1], dtype=np.uint16), if_scene=False).tree
    for node in lt.iter_search_nodes():
        if node.name != -1 and isinstance(node.name, (int, float)):
            up_name = node.up.name + 1 if isinstance(node.up.name, (int, float)) else 0
            lineage.append([node.name + 1, node.frame, int(node.frame+node.dist-1), up_name])
            track.append(node.track)
    # our cell ids start from -1 (missing cell or root node), but CTC's start from 0
    # lineage = np.array(lineage)
    if not saveRGB:
        np.savetxt(os.path.join(save_path, res_path, "res_track.txt"), lineage, fmt="%d")
    # # store lineage

    # generate random colors for each cell
    np.random.seed(12345)
    RGB = np.random.randint(0, 255, size=(len(lineage), 3))
    # mask_dir = np.sort(glob(abs_path(join(mask, '*.tif')))).tolist()[: tracks.shape[0]]
    mask_files = np.sort(os.listdir(mask_dir))[: tracks.shape[0]]
    for frame, mask_file in enumerate(mask_files):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = tifffile.imread(mask_path)

        # generate tracking mask in CTC format
        # track_img = np.zeros_like(mask, dtype=np.uint16)
        track_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint16) \
            if saveRGB else np.zeros_like(mask, dtype=np.uint16)
        for c, cell in enumerate(lineage):
            if frame >= cell[1] and frame <= cell[2]:
                # if track[c][frame-cell[1]] != -1:
                # coord = centroid[frame][track[c][frame-cell[1]]][0].astype(np.int16)
                # id = mask[coord[0], coord[1]]
                cell_local = mask == centroid[frame][track[c][frame-cell[1]]][0, 2].astype(np.int16)
                # #空心细胞变成实心细胞
                # if np.sum(cell_local) > mask.size / 1000:
                #     #生成一个圆mask
                #     x, y = np.indices(mask.shape)
                #     cell_local = (x - coord[0])**2 + (y - coord[1])**2 < 36
                # track_img[cell_local] = cell[0]  # RGB[c]
                track_img[cell_local] = RGB[c] if saveRGB else cell[0]
                # else:
                #     mask_path = os.path.join(save_path, mask_files[frame - 1])
                #     mask_pref = tifffile.imread(mask_path)
                #     track_img[mask_pref == id] = cell[0]  # RGB[c]
        if saveRGB:
            tifffile.imwrite(os.path.join(save_path, 'rgb'+res_path, mask_file), track_img)
        else:
            tifffile.imwrite(os.path.join(save_path, res_path, mask_file), track_img)
    if saveRGB:
        save_as_gif(os.path.join(save_path, 'rgb'+res_path))


def save_as_gif(save_dir):
    from PIL import Image
    import os

    # # 图片文件夹路径（请修改为你的实际路径）
    # image_folder = 'path_to_your_images'  # 例如 'frames/'
    # 输出的 gif 文件名
    output_gif = 'track_mask.gif'

    # 获取该目录下所有图片文件，并按文件名排序（确保顺序正确！）
    images = sorted([img for img in os.listdir(save_dir) if img.endswith(".tif")])

    # 打开第一张图片，用于获取尺寸，并作为 GIF 的第一帧
    frames = []
    for image_name in images:
        frame_path = os.path.join(save_dir, image_name)
        frame = tifffile.imread(frame_path).astype(np.uint8)
        frames.append(Image.fromarray(frame))

    # 保存为 GIF
    # 参数说明：
    # save_all=True 表示保存多帧
    # append_images=frames[1:] 表示从第二帧开始追加
    # duration=200 表示每帧显示 200 毫秒（可调整，控制播放速度）
    # loop=0 表示无限循环，设为数字比如 3 就是循环 3 次
    frames[0].save(
        os.path.join(save_dir, output_gif),
        save_all=True,
        append_images=frames[1:],
        duration=50,  # 每帧持续时间（毫秒）
        loop=0  # 循环次数，0 为无限循环
    )

    print(f"GIF 已保存为：{output_gif}")


if __name__ == "__main__":
    # mask_dir = r"K:\celltrack\stardist"
    # track_dir = r"I:\CellTrack\result\dataset1\test_0_50_frame\2025-06-20track_linear_solver_merged_and_pruned_50.csv"
    # centroid = r"I:\CellTrack\result\dataset1\test_0_50_frame\2025-05-26-centroid.npy"

    mask_dir = r"I:\CellTrack\dataset\Fluo-N2DH-GOWT\01_ST\SEG"
    track_dir = r"I:\CellTrack\experiment\gowt1\result\2025-08-11track_linear_solver_merged_and_pruned.csv"
    centroid = r"I:\CellTrack\experiment\gowt1\result\2025-08-11-centroid.npy"

    tracks = pd.read_csv(track_dir).to_numpy()
    cnt = np.load(centroid, allow_pickle=True)
    lineage_mask(mask_dir, tracks, cnt, save_path=r"I:\CellTrack\dataset\Fluo-N2DH-GOWT", res_path='RES1')
    lineage_mask(mask_dir, tracks, cnt, save_path=r"I:\CellTrack\dataset\Fluo-N2DH-GOWT", res_path='RES1', saveRGB=True)

    # pp = pprint.PrettyPrinter(indent=4)
    #
    # gt_data = load_ctc_data(
    #     r"I:\CellTrack\dataset\Fluo-N2DL-HeLa\01_GT\TRA",
    #     r"I:\CellTrack\dataset\Fluo-N2DL-HeLa\01_GT\TRA\man_track.txt",
    #     name="Hela-01_GT",
    # )
    # pred_data = load_ctc_data(
    #     r"I:\CellTrack\dataset\Fluo-N2DL-HeLa\01_GT\TRA",
    #     r"I:\CellTrack\dataset\Fluo-N2DL-HeLa\01_GT\TRA\man_track.txt",
    #     name="Hela-01_RES",
    # )
    #
    # # ctc_results, ctc_matched = run_metrics(
    # #     gt_data=gt_data,
    # #     pred_data=pred_data,
    # #     matcher=CTCMatcher(),
    # #     metrics=[CTCMetrics()]
    # # )
    #
    # ctc_results = run_metrics(
    #     gt_data=gt_data,
    #     pred_data=pred_data,
    #     matcher=CTCMatcher(),
    #     metrics=[CTCMetrics()],
    # )
    # pp.pprint(ctc_results)
    # # link = ctc_results["LNK"]
    #
    # iou_results = run_metrics(
    #     gt_data=gt_data,
    #     pred_data=pred_data,
    #     matcher=IOUMatcher(iou_threshold=0.1),
    #     metrics=[DivisionMetrics(max_frame_buffer=2)],
    # )
    # pp.pprint(iou_results)
    #
    # print('Done!')
