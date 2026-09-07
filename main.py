
if __name__ == '__main__':
    from train import train_model as tm
    from predictor import predictor as pr
    from cell_tracking import tracker as ct
    import datetime
    import os.path.join as join
    from hydra.utils import to_absolute_path as abs_path

    now = datetime.datetime.now()
    print('START!', now)

    tm()
    pr()
    ct()

    now = datetime.datetime.now()
    print('track DONE!', now)

    import pandas as pd
    import numpy as np
    from convert2CTCFormat import lineage_mask

    mask_dir = abs_path(join('data', 'mask'))
    track_dir = abs_path(join("result", "track_results.csv"))
    centroid = abs_path(join("result", "centroid.npy"))

    tracks = pd.read_csv(track_dir).to_numpy()
    cnt = np.load(centroid, allow_pickle=True)
    lineage_mask(mask_dir, tracks, cnt, save_path=r"result", res_path='RES1')
    lineage_mask(mask_dir, tracks, cnt, save_path=r"result", res_path='RES1', saveRGB=True)
